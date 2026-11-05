"""
lap_learning_studio.py
======================
CarRacing-v3 Deep Reinforcement Learning Studio:
- Whole View of the Lap (Full Circuit Minimap)
- Off-Road Penalty & Instant Restart from Start Line
- Real-time Lap Stopwatch & Finish Line Time Tracking
- Step-by-Step Learning Progression Recording
- Automated Video Compilation (lap_learning_journey.mp4 & lap_winning_run.mp4)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import gymnasium as gym
import imageio
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RECORDINGS_DIR = PROJECT_ROOT / "recorded_races"

CHECKPOINT_DIR.mkdir(exist_ok=True)
RECORDINGS_DIR.mkdir(exist_ok=True)


# ==============================================================================
# 1. State Preprocessing & Wrappers
# ==============================================================================

def preprocess_frame(img: np.ndarray) -> np.ndarray:
    """Crops observation to 60x60 grayscale normalized float32."""
    crop = img[24:84, 17:77]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return gray.astype(np.float32) / 255.0


class OffRoadPenaltyEnv(gym.Wrapper):
    """
    Environment wrapper that enforces:
    - Off-asphalt detection: if wheels leave the road, applies penalty and restarts immediately.
    - Lap progress measurement & finish line detection.
    - Elapsed simulation time tracking.
    """
    # Minimum steps before off-road penalty can fire (grace period for episode start zoom animation)
    OFFROAD_GRACE_STEPS = 40

    def __init__(
        self,
        env: gym.Env,
        penalty: float = 100.0,
        initial_no_op: int = 0,
        skip_frames: int = 2,
        stack_frames: int = 4,
    ):
        super().__init__(env)
        self.penalty = penalty
        self.initial_no_op = initial_no_op
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames

        # Session-level counters (never reset between episodes)
        self.restarts = 0            # total off-road events this session
        self.total_laps_finished = 0 # total lap completions this session
        self.crashes: list[tuple[float, float]] = []  # all crash positions

        # Episode-level counters (reset each episode)
        self.step_count = 0
        self.episode_offroads = 0    # off-road events in current episode
        self.best_progress = 0.0
        self.lap_finished = False
        self.finish_time: float | None = None
        self.current_trail: list[tuple[float, float]] = []
        self.stacked_state = np.zeros((stack_frames, 60, 60), dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Reset episode-level state only
        self.step_count = 0
        self.episode_offroads = 0
        self.current_trail = []
        self.lap_finished = False
        self.finish_time = None
        # Note: self.restarts, self.crashes, self.total_laps_finished persist across episodes

        car = self.unwrapped.car
        if car is not None and hasattr(car, "hull"):
            self.current_trail.append((float(car.hull.position.x), float(car.hull.position.y)))

        # Warmup steps if requested — advances physics so tile contacts register
        for _ in range(self.initial_no_op):
            obs, _, _, _, info = self.env.step(0)
            car = self.unwrapped.car
            if car is not None and hasattr(car, "hull"):
                self.current_trail.append((float(car.hull.position.x), float(car.hull.position.y)))

        s = preprocess_frame(obs)
        self.stacked_state = np.tile(s, (self.stack_frames, 1, 1))
        return self.stacked_state, info

    def step(self, action: int):
        reward_total = 0.0
        terminated = False
        truncated = False
        info = {}
        obs = None  # ensure obs is always bound

        for _ in range(self.skip_frames):
            obs, reward, term, trunc, step_info = self.env.step(action)
            reward_total += reward
            self.step_count += 1
            info.update(step_info)

            car = self.unwrapped.car

            # Guard: car must exist and have wheels before any attribute access
            if car is None or not hasattr(car, "hull") or not hasattr(car, "wheels"):
                if term or trunc:
                    terminated, truncated = term, trunc
                    break
                continue

            # Record breadcrumb trail
            self.current_trail.append((float(car.hull.position.x), float(car.hull.position.y)))

            # Count wheels touching road tiles
            try:
                on_road_wheels = sum(1 for w in car.wheels if len(w.tiles) > 0)
            except Exception:
                on_road_wheels = 4  # assume on-road if check fails
            info["on_road_wheels"] = on_road_wheels

            # Track progress
            total_tiles = max(1, len(self.unwrapped.track))
            progress = min(100.0, (self.unwrapped.tile_visited_count / total_tiles) * 100.0)
            info["lap_progress"] = progress
            self.best_progress = max(self.best_progress, progress)

            # Elapsed time: each environment step is 0.02s
            elapsed = self.step_count * 0.02
            info["elapsed_time"] = elapsed

            # 1. Off-Road Rule: penalty fires only after grace period
            #    (first OFFROAD_GRACE_STEPS steps are immune — covers zoom-in animation
            #    during which Box2D tile contacts haven't fully registered yet)
            past_grace = self.step_count > self.OFFROAD_GRACE_STEPS
            if past_grace and on_road_wheels == 0:
                reward_total -= self.penalty
                terminated = True
                info["off_road"] = True
                self.restarts += 1          # session total
                self.episode_offroads += 1  # episode total
                crash_pos = (float(car.hull.position.x), float(car.hull.position.y))
                self.crashes.append(crash_pos)
                info["crash_pos"] = crash_pos
                info["session_restarts"] = self.restarts
                info["episode_offroads"] = self.episode_offroads
                break

            # 2. Finish Line Rule: Lap complete or > 95% track tiles visited
            is_complete = bool(info.get("lap_complete", False) or progress >= 95.0)
            if is_complete:
                self.lap_finished = True
                self.total_laps_finished += 1
                if self.finish_time is None:
                    self.finish_time = elapsed
                info["lap_finished"] = True
                info["finish_time"] = self.finish_time
                info["total_laps_finished"] = self.total_laps_finished
                reward_total += 200.0  # Big bonus for finishing
                terminated = True
                break

            # NOTE: We intentionally do NOT break on inner env's term/trunc here.
            # Episodes must only end via off-road penalty or lap completion above.
            # The gymnasium TimeLimit wrapper is stripped at env creation time.

        # Safety: if obs was never set (edge case: skip_frames=0), get a fresh render
        if obs is None:
            obs = self.env.render() if hasattr(self.env, "render") else np.zeros((96, 96, 3), np.uint8)

        # Always forward episode-level off-road count in info
        info["episode_offroads"] = self.episode_offroads
        info["session_restarts"] = self.restarts

        s = preprocess_frame(obs)
        self.stacked_state = np.concatenate((self.stacked_state[1:], s[np.newaxis]), axis=0)
        return self.stacked_state, reward_total, terminated, truncated, info


# ==============================================================================
# 2. Whole View Lap Renderer & Telemetry Composite
# ==============================================================================

class WholeLapRenderer:
    """
    Renders a unified 1120x640 composite view:
    - Left (640x640): High-resolution driving camera + active status banner.
    - Right (480x640):
        * Top: "WHOLE VIEW OF THE LAP" (full circuit overview, car dot, trail, crash marks).
        * Bottom: Telemetry, stopwatch timer, progress bar, off-road penalty stats.
    """
    def __init__(self, env: gym.Env, canvas_width: int = 1120, canvas_height: int = 640):
        self.env = env
        self.width = canvas_width
        self.height = canvas_height
        self.map_w = 440
        self.map_h = 410

        self.min_x = 0.0
        self.max_x = 1.0
        self.min_y = 0.0
        self.max_y = 1.0
        self.scale = 1.0
        self.pad_x = 0.0
        self.pad_y = 0.0

        self._calibrate_track_bounds()
        self.cached_base_map = self._build_base_map()

    def update_track(self, env: gym.Env):
        """Re-calibrates track bounds and rebuilds base map whenever environment resets."""
        self.env = env
        self._calibrate_track_bounds()
        self.cached_base_map = self._build_base_map()

    def _calibrate_track_bounds(self):
        unwrapped = self.env.unwrapped
        all_pts = [pt for poly, _ in unwrapped.road_poly for pt in poly]
        all_pts = np.array(all_pts)
        self.min_x, self.min_y = all_pts.min(axis=0)
        self.max_x, self.max_y = all_pts.max(axis=0)

        span_x = max(1.0, self.max_x - self.min_x)
        span_y = max(1.0, self.max_y - self.min_y)

        self.scale = min((self.map_w - 40) / span_x, (self.map_h - 40) / span_y)
        self.pad_x = (self.map_w - span_x * self.scale) / 2
        self.pad_y = (self.map_h - span_y * self.scale) / 2

    def world_to_map(self, x: float, y: float) -> tuple[int, int]:
        px = int((x - self.min_x) * self.scale + self.pad_x)
        py = int(self.map_h - ((y - self.min_y) * self.scale + self.pad_y))
        return max(0, min(self.map_w - 1, px)), max(0, min(self.map_h - 1, py))

    def _build_base_map(self) -> np.ndarray:
        unwrapped = self.env.unwrapped
        base = np.zeros((self.map_h, self.map_w, 3), dtype=np.uint8)
        # Sleek dark track grass
        base[:] = (20, 32, 22)

        # Draw road polygons
        for poly, color in unwrapped.road_poly:
            pts = np.array([self.world_to_map(x, y) for x, y in poly], dtype=np.int32)
            # Differentiate curb vs road
            if color[0] == color[1] == color[2]:
                c = (70, 70, 75)  # dark asphalt
            else:
                c = (int(color[2]), int(color[1]), int(color[0]))  # curbs
            cv2.fillPoly(base, [pts], c)

        # Draw Checkered Start / Finish Line
        if len(unwrapped.track) > 0:
            s_tile = unwrapped.track[0]
            sx, sy = self.world_to_map(s_tile[2], s_tile[3])
            cv2.circle(base, (sx, sy), 8, (255, 255, 255), -1)
            cv2.circle(base, (sx, sy), 6, (0, 0, 0), -1)
            cv2.circle(base, (sx, sy), 3, (255, 255, 255), -1)
            cv2.putText(base, "FINISH LINE", (max(10, sx - 42), max(15, sy - 10)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.38, (0, 240, 255), 1)

        return base

    def render_composite(
        self,
        info: dict,
        episode: int,
        reward: float,
        best_time: float | None = None,
        status_text: str = "RACING",
        status_color: tuple[int, int, int] = (0, 220, 100),
        speed_text: str = "1.0x",
        epsilon: float = 1.0,
    ) -> np.ndarray:
        # 1. Main Camera View (640x640)
        cam = self.env.render()
        cam = cv2.resize(cam, (640, 640))

        # Canvas initialization
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:, :640] = cam
        canvas[:, 640:] = (18, 22, 28)  # Modern dark charcoal dashboard

        # 2. Render Whole View Map
        minimap = self.cached_base_map.copy()
        unwrapped = self.env.unwrapped
        wrapper: OffRoadPenaltyEnv = self.env

        # Draw crash points (historical off-road penalties)
        for cx, cy in wrapper.crashes:
            mx, my = self.world_to_map(cx, cy)
            cv2.drawMarker(minimap, (mx, my), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=8, thickness=2)

        # Draw breadcrumb trail of current lap
        if len(wrapper.current_trail) > 1:
            pts = np.array([self.world_to_map(x, y) for x, y in wrapper.current_trail], dtype=np.int32)
            cv2.polylines(minimap, [pts], isClosed=False, color=(0, 240, 255), thickness=2)

        # Draw current car with true orientation and headlight beam
        car = unwrapped.car
        if car and hasattr(car, "hull"):
            car_x, car_y = car.hull.position.x, car.hull.position.y
            mx, my = self.world_to_map(car_x, car_y)

            angle = car.hull.angle
            # Box2D car forward vector: (-sin(angle), cos(angle)). Inverted screen Y:
            fx = -math.sin(angle)
            fy = -math.cos(angle)
            rx = math.cos(angle)
            ry = -math.sin(angle)

            # Headlight beam cone on track
            cone_len = 20
            cone_w = 9
            c_p1 = (mx, my)
            c_p2 = (int(mx + cone_len * fx + cone_w * rx), int(my + cone_len * fy + cone_w * ry))
            c_p3 = (int(mx + cone_len * fx - cone_w * rx), int(my + cone_len * fy - cone_w * ry))
            cv2.fillPoly(minimap, [np.array([c_p1, c_p2, c_p3], dtype=np.int32)], (60, 180, 250))

            # Car chassis polygon
            car_len = 8.0
            car_w = 4.0
            b1 = (int(mx + car_len * fx + car_w * rx), int(my + car_len * fy + car_w * ry))
            b2 = (int(mx + car_len * fx - car_w * rx), int(my + car_len * fy - car_w * ry))
            b3 = (int(mx - car_len * fx - car_w * rx), int(my - car_len * fy - car_w * ry))
            b4 = (int(mx - car_len * fx + car_w * rx), int(my - car_len * fy + car_w * ry))
            car_poly = np.array([b1, b2, b3, b4], dtype=np.int32)

            cv2.fillPoly(minimap, [car_poly], (0, 30, 240))  # Racing red car
            cv2.polylines(minimap, [car_poly], isClosed=True, color=(255, 255, 255), thickness=1)

            # Center position glow
            cv2.circle(minimap, (mx, my), 3, (0, 255, 255), -1)

        # Place minimap onto canvas
        canvas[20:20 + self.map_h, 660:660 + self.map_w] = minimap

        # Map Title Header
        cv2.putText(canvas, "WHOLE VIEW OF THE LAP", (745, 15),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, (200, 220, 240), 1)

        # 3. Telemetry & Dashboard Panel (Bottom Right)
        panel_y = 20 + self.map_h + 15

        elapsed = info.get("elapsed_time", 0.0)
        time_str = f"{int(elapsed // 60):02d}:{elapsed % 60:05.2f}"
        progress = info.get("lap_progress", 0.0)
        session_restarts = info.get("session_restarts", wrapper.restarts)
        episode_offroads = info.get("episode_offroads", wrapper.episode_offroads)
        total_laps = getattr(wrapper, "total_laps_finished", 0)

        # Status Badge
        cv2.rectangle(canvas, (660, panel_y), (1120, panel_y + 35), (28, 34, 44), -1)
        cv2.rectangle(canvas, (660, panel_y), (660 + 8, panel_y + 35), status_color, -1)
        cv2.putText(canvas, f"STATUS: {status_text}", (680, panel_y + 24),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, status_color, 1)

        # Row 1: Timer + Best Finish
        row1_y = panel_y + 52
        cv2.putText(canvas, f"LAP TIMER : {time_str}", (660, row1_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        best_str = f"{int(best_time // 60):02d}:{best_time % 60:05.2f}" if best_time else "--:--"
        cv2.putText(canvas, f"BEST FINISH: {best_str}", (895, row1_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.46, (0, 220, 255), 1)

        # Row 2: Episode + Lap progress
        row2_y = row1_y + 23
        cv2.putText(canvas, f"EPISODE   : #{episode}", (660, row2_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.46, (200, 200, 200), 1)
        cv2.putText(canvas, f"LAPS DONE : {total_laps}", (895, row2_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.46, (0, 255, 160), 1)

        # Row 3: Off-Road counters — per-episode and session total (the key fix)
        row3_y = row2_y + 23
        offroad_ep_color = (255, 80, 80) if episode_offroads > 0 else (140, 140, 155)
        offroad_sess_color = (255, 140, 40) if session_restarts > 0 else (140, 140, 155)
        cv2.putText(canvas, f"OFF-ROAD EP: {episode_offroads}", (660, row3_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, offroad_ep_color, 1)
        cv2.putText(canvas, f"TOTAL SESSION: {session_restarts}", (840, row3_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.48, offroad_sess_color, 1)

        # Row 4: Sim Speed + Epsilon
        row4_y = row3_y + 23
        speed_color = (0, 255, 255) if "TURBO" in speed_text else ((255, 120, 120) if "PAUSE" in speed_text else (100, 255, 160))
        cv2.putText(canvas, f"SIM SPEED : {speed_text}", (660, row4_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.46, speed_color, 1)
        eps_color = (255, 200, 0) if epsilon > 0.5 else ((100, 255, 100) if epsilon < 0.15 else (200, 200, 200))
        cv2.putText(canvas, f"EPSILON: {epsilon:.3f}", (895, row4_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.46, eps_color, 1)

        # Row 5: Progress Bar
        row5_y = row4_y + 23
        cv2.putText(canvas, f"LAP PROGRESS: {progress:.1f}%", (660, row5_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.46, (255, 220, 100), 1)
        cv2.putText(canvas, "[+/- Speed | SPACE Pause | M Turbo | R 1x]", (660, row5_y + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (130, 145, 165), 1)

        bar_x, bar_y = 660, row5_y + 22
        bar_w, bar_h = 440, 11
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 48, 60), -1)
        fill_w = int((progress / 100.0) * bar_w)
        if fill_w > 0:
            fill_color = (0, 220, 120) if progress < 90 else (0, 255, 255)
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), fill_color, -1)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 95, 115), 1)

        # Left Camera HUD Banner — now includes off-road stats
        cv2.rectangle(canvas, (8, 8), (500, 42), (0, 0, 0, 180), -1)
        hud_offroad = f"OFF-ROAD ep:{episode_offroads} tot:{session_restarts}"
        cv2.putText(canvas, f"TIME {time_str} | REW {reward:+.1f} | {hud_offroad}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)

        return canvas


# ==============================================================================
# 3. Progressive Deep Q-Network Agent
# ==============================================================================

class CNNActionValue(nn.Module):
    def __init__(self, state_dim: int = 4, action_dim: int = 5):
        super().__init__()
        self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.in_features = 32 * 6 * 6
        self.fc1 = nn.Linear(self.in_features, 256)
        self.fc2 = nn.Linear(256, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view((-1, self.in_features))
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class ReplayBuffer:
    def __init__(self, state_dim: tuple[int, ...], max_size: int = 50000):
        self.s = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.a = np.zeros((max_size, 1), dtype=np.int64)
        self.r = np.zeros((max_size, 1), dtype=np.float32)
        self.s_prime = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.terminated = np.zeros((max_size, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.max_size = max_size

    def update(self, s, a, r, s_prime, terminated):
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.s_prime[self.ptr] = s_prime
        self.terminated[self.ptr] = float(terminated)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.s[idx]),
            torch.LongTensor(self.a[idx]),
            torch.FloatTensor(self.r[idx]),
            torch.FloatTensor(self.s_prime[idx]),
            torch.FloatTensor(self.terminated[idx]),
        )


class ProgressiveAgent:
    def __init__(
        self,
        state_dim: tuple[int, ...] = (4, 60, 60),
        action_dim: int = 5,
        lr: float = 0.0003,
        gamma: float = 0.99,
        batch_size: int = 32,
        buffer_size: int = 50000,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.992
        self.total_steps = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = CNNActionValue(state_dim[0], action_dim).to(self.device)
        self.target_network = CNNActionValue(state_dim[0], action_dim).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = ReplayBuffer(state_dim, max_size=buffer_size)

    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        with torch.no_grad():
            x = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q = self.network(x)
            return int(torch.argmax(q).item())

    def learn(self):
        if self.buffer.size < self.batch_size:
            return None
        s, a, r, s_prime, term = [x.to(self.device) for x in self.buffer.sample(self.batch_size)]

        with torch.no_grad():
            next_q = self.target_network(s_prime)
            td_target = r + (1.0 - term) * self.gamma * next_q.max(dim=1, keepdim=True).values

        q_val = self.network(s).gather(1, a)
        loss = F.mse_loss(q_val, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.total_steps % 1000 == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        return loss.item()

    def step_decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ==============================================================================
# 4. Video Recording & Stitching
# ==============================================================================

def stitch_learning_journey(video_paths: list[Path], output_path: Path, fps: int = 25):
    """Combines all milestone video clips into a single progression movie."""
    print(f"\n[Stitching] Compiling {len(video_paths)} milestone clips into {output_path.name}...")
    writer = imageio.get_writer(str(output_path), fps=fps)
    for p in video_paths:
        if not p.exists():
            continue
        reader = imageio.get_reader(str(p))
        for frame in reader:
            writer.append_data(frame)
        reader.close()
    writer.close()
    print(f"[Done] Learning journey video ready: {output_path}")


# ==============================================================================
# 5. Main Training & Simulation Loop
# ==============================================================================

def run_learning_studio(
    total_episodes: int = 5000,
    render_gui: bool = False,
    off_road_penalty: float = 100.0,
    save_milestone_every: int = 10,
    resume_checkpoint: bool = False,
    track_seed: int = 42,
    initial_speed: float = 1.0,
):
    print("\n" + "=" * 65)
    print("   [CAR RACING v3: STEP-BY-STEP LEARNING STUDIO]")
    print("=" * 65)
    print(f"Target Episodes   : {total_episodes}")
    print(f"Track Circuit Seed: {track_seed} (Fixed path for consistent learning)")
    print(f"Off-Road Penalty  : -{off_road_penalty:.1f} (triggers immediate restart)")
    print(f"Initial Sim Speed : {'TURBO (MAX)' if initial_speed == 0 else f'{initial_speed:.1f}x'}")
    print(f"Live GUI Window   : {'Enabled' if render_gui else 'Disabled (Headless background)'}")
    print(f"Live Hotkeys      : [ + / UP ] Speed Up | [ - / DOWN ] Slow Down")
    print(f"                    [ SPACE ] Pause/Resume | [ M ] Turbo Max | [ R ] Reset 1x")
    print(f"Output Directory  : {RECORDINGS_DIR}\n")

    base_env = gym.make("CarRacing-v3", domain_randomize=False, continuous=False, render_mode="rgb_array")
    # Strip gymnasium's built-in TimeLimit wrapper — episodes now only end on
    # off-road penalty or lap completion, not on a fixed step count.
    while isinstance(base_env, gym.wrappers.TimeLimit):
        base_env = base_env.env
    env = OffRoadPenaltyEnv(base_env, penalty=off_road_penalty)
    # Warm reset to initialize track bounds and road polygons
    initial_state, initial_info = env.reset(seed=track_seed)
    renderer = WholeLapRenderer(env)

    agent = ProgressiveAgent(state_dim=(4, 60, 60), action_dim=5)

    dqn_weights = (MODELS_DIR / "true_dqn.pt") if (MODELS_DIR / "true_dqn.pt").exists() else (PROJECT_ROOT / "true_dqn.pt")
    if resume_checkpoint and dqn_weights.exists():
        print(f">> Loading starting weights from {dqn_weights.name}...")
        agent.network.load_state_dict(torch.load(dqn_weights, map_location=agent.device))
        agent.target_network.load_state_dict(agent.network.state_dict())
        agent.epsilon = 0.35  # Moderate exploration for fine-tuning

    # Speed control presets
    SPEED_PRESETS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0, 0.0]
    speed_idx = 2  # default 1.0x
    for i, sp in enumerate(SPEED_PRESETS):
        if abs(sp - initial_speed) < 0.05:
            speed_idx = i
            break
        elif initial_speed == 0:
            speed_idx = len(SPEED_PRESETS) - 1
            break
    sim_speed = SPEED_PRESETS[speed_idx]
    is_paused = False

    # Pygame setup if rendering live
    screen = None
    clock = None
    if render_gui:
        pygame.init()
        screen = pygame.display.set_mode((1120, 640))
        pygame.display.set_caption("CarRacing-v3 — Whole View & Learning Telemetry")
        clock = pygame.time.Clock()

    milestone_videos: list[Path] = []
    best_time: float | None = None
    best_lap_progress = 0.0

    for ep in range(1, total_episodes + 1):
        state, info = env.reset(seed=track_seed)
        renderer.update_track(env)
        done = False
        episode_reward = 0.0
        step = 0

        # Determine if this is a pre-planned milestone episode
        is_milestone = (ep == 1) or (ep % save_milestone_every == 0) or (ep == total_episodes)
        # Always collect frames in a rolling buffer — we save for off-road/finish too
        # deque(maxlen=1800) caps memory at ~1800 frames regardless of episode length
        frames_to_record: deque[np.ndarray] = deque(maxlen=1800)
        episode_had_offroad = False

        status_text = "LEARNING"
        status_color = (0, 220, 100)

        while not done:
            action = agent.act(state, training=True)
            next_state, reward, term, trunc, info = env.step(action)
            done = term or trunc

            agent.buffer.update(state, action, reward, next_state, done)
            agent.learn()
            agent.total_steps += 1

            state = next_state
            episode_reward += reward
            step += 1

            # Check status flags
            if info.get("off_road", False):
                status_text = "OFF-ROAD RESTART! PENALTY -100"
                status_color = (0, 0, 255)
                episode_had_offroad = True
            elif info.get("lap_finished", False):
                status_text = f"FINISH LINE CROSSED in {info['finish_time']:.2f}s!"
                status_color = (0, 255, 255)
                if best_time is None or info["finish_time"] < best_time:
                    best_time = info["finish_time"]

            # Speed text formatting
            if is_paused:
                speed_str = "PAUSED"
            elif sim_speed == 0.0:
                speed_str = "TURBO (MAX)"
            else:
                speed_str = f"{sim_speed:.1f}x"

            # Frame skip rendering optimization for ultra-high speed acceleration
            should_blit = True
            if sim_speed >= 4.0 and (step % 2 != 0):
                should_blit = False
            elif (sim_speed >= 8.0 or sim_speed == 0.0) and (step % 4 != 0):
                should_blit = False

            # Always render composite frame (needed for rolling recording buffer).
            # Display to screen only when should_blit is True (speed optimization).
            if True:
                comp_frame = renderer.render_composite(
                    info=info,
                    episode=ep,
                    reward=episode_reward,
                    best_time=best_time,
                    status_text=status_text,
                    status_color=status_color,
                    speed_text=speed_str,
                    epsilon=agent.epsilon,
                )

                # Always collect into rolling buffer for potential video save
                frames_to_record.append(comp_frame)

                # Pygame GUI display
                if render_gui and screen is not None and should_blit:
                    surf = pygame.surfarray.make_surface(np.transpose(comp_frame, (1, 0, 2)))
                    screen.blit(surf, (0, 0))
                    pygame.display.flip()

            # Pygame event handling & interactive speed adjustment
            if render_gui and screen is not None:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("\nSimulation aborted by user.")
                        pygame.quit()
                        env.close()
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_q, pygame.K_ESCAPE):
                            print("\nExiting live window...")
                            pygame.quit()
                            env.close()
                            return
                        elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_UP):
                            speed_idx = min(len(SPEED_PRESETS) - 1, speed_idx + 1)
                            sim_speed = SPEED_PRESETS[speed_idx]
                            sp_name = 'TURBO (MAX)' if sim_speed == 0 else f'{sim_speed:.1f}x'
                            print(f">> [Speed] Accelerated to: {sp_name}")
                        elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS, pygame.K_DOWN):
                            speed_idx = max(0, speed_idx - 1)
                            sim_speed = SPEED_PRESETS[speed_idx]
                            sp_name = 'TURBO (MAX)' if sim_speed == 0 else f'{sim_speed:.1f}x'
                            print(f">> [Speed] Decelerated to: {sp_name}")
                        elif event.key == pygame.K_m:
                            if sim_speed == 0.0:
                                speed_idx = 2  # return to 1.0x
                                sim_speed = 1.0
                            else:
                                speed_idx = len(SPEED_PRESETS) - 1
                                sim_speed = 0.0
                            sp_name = 'TURBO (MAX)' if sim_speed == 0 else f'{sim_speed:.1f}x'
                            print(f">> [Speed] Turbo toggled: {sp_name}")
                        elif event.key == pygame.K_r:
                            speed_idx = 2
                            sim_speed = 1.0
                            print(">> [Speed] Reset to 1.0x")
                        elif event.key == pygame.K_SPACE:
                            is_paused = not is_paused
                            print(f">> [Simulation] {'PAUSED' if is_paused else 'RESUMED'}")

                # Pause loop
                while is_paused:
                    comp_frame = renderer.render_composite(
                        info=info,
                        episode=ep,
                        reward=episode_reward,
                        best_time=best_time,
                        status_text="PAUSED",
                        status_color=(255, 180, 0),
                        speed_text="PAUSED",
                        epsilon=agent.epsilon,
                    )
                    surf = pygame.surfarray.make_surface(np.transpose(comp_frame, (1, 0, 2)))
                    screen.blit(surf, (0, 0))
                    pygame.display.flip()

                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            env.close()
                            return
                        elif event.type == pygame.KEYDOWN:
                            if event.key in (pygame.K_q, pygame.K_ESCAPE):
                                pygame.quit()
                                env.close()
                                return
                            elif event.key == pygame.K_SPACE:
                                is_paused = False
                                print(">> [Simulation] RESUMED")
                                break
                    if clock:
                        clock.tick(15)

                if clock:
                    if sim_speed == 0.0:
                        clock.tick(0)
                    else:
                        target_fps = max(1, int(30 * sim_speed))
                        clock.tick(target_fps)

        agent.step_decay()
        prog = info.get("lap_progress", 0.0)
        best_lap_progress = max(best_lap_progress, prog)
        frames_list = list(frames_to_record)  # snapshot deque once

        # Print episode summary
        flag_status = (
            "[LAP FINISHED!]" if info.get("lap_finished")
            else ("[OFF-ROAD RESTART]" if info.get("off_road") else "[RUNNING->DONE]")
        )
        print(f"Ep {ep:3d}/{total_episodes:3d} | Steps: {step:4d} | Reward: {episode_reward:7.1f} | "
              f"Progress: {prog:5.1f}% | Epsilon: {agent.epsilon:.3f} | {flag_status}")

        # Save milestone video (pre-planned periodic episodes)
        if is_milestone and frames_list:
            vid_path = RECORDINGS_DIR / f"milestone_ep{ep:03d}.mp4"
            imageio.mimsave(str(vid_path), frames_list, fps=25)
            milestone_videos.append(vid_path)
            print(f"   [VIDEO] Saved milestone clip: {vid_path.name}")

        # Save off-road episode clip (every time car goes off-road)
        elif episode_had_offroad and frames_list:
            vid_path = RECORDINGS_DIR / f"offroad_ep{ep:03d}.mp4"
            imageio.mimsave(str(vid_path), frames_list, fps=25)
            milestone_videos.append(vid_path)
            print(f"   [OFF-ROAD VIDEO] Saved off-road clip: {vid_path.name} "
                  f"(restarts this session: {env.restarts})")

        # If lap was finished, record dedicated winning lap video
        if info.get("lap_finished", False) and frames_list:
            win_vid = RECORDINGS_DIR / "lap_winning_run.mp4"
            imageio.mimsave(str(win_vid), frames_list, fps=25)
            print(f"   [VICTORY] Winning lap saved: {win_vid.name} (Time: {info['finish_time']:.2f}s)!")

    # Checkpoint save
    torch.save(agent.network.state_dict(), CHECKPOINT_DIR / "progressive_dqn_latest.pt")
    print(f"\nSaved trained model weights to {CHECKPOINT_DIR / 'progressive_dqn_latest.pt'}")

    # Compile master learning journey video
    if milestone_videos:
        master_vid = RECORDINGS_DIR / "lap_learning_journey.mp4"
        stitch_learning_journey(milestone_videos, master_vid, fps=25)

    if render_gui:
        pygame.quit()
    env.close()

    print("\n" + "=" * 65)
    print("   LEARNING STUDIO COMPLETED!")
    print(f"   Best Lap Progress : {best_lap_progress:.1f}%")
    if best_time:
        print(f"   Best Finish Time  : {best_time:.2f} seconds")
    print(f"   Master Video      : {RECORDINGS_DIR / 'lap_learning_journey.mp4'}")
    print("=" * 65)


# ==============================================================================
# 6. CLI Entry Point
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CarRacing-v3 Learning Studio with Whole Lap View & Off-Road Penalties")
    parser.add_argument("--episodes", type=int, default=50, help="Total episodes to train (default: 50)")
    parser.add_argument("--render", action="store_true", help="Display live interactive Pygame window")
    parser.add_argument("--penalty", type=float, default=100.0, help="Penalty for going off asphalt (default: 100)")
    parser.add_argument("--save-every", type=int, default=10, help="Save milestone video every N episodes (default: 10)")
    parser.add_argument("--resume", action="store_true", help="Resume from reference weights")
    parser.add_argument("--track-seed", type=int, default=42, help="Seed for fixed circuit track layout (default: 42)")
    parser.add_argument("--speed", type=float, default=1.0, help="Initial simulation speed multiplier (e.g. 1.0, 2.0, 5.0, or 0 for TURBO MAX, default: 1.0)")
    args = parser.parse_args()

    run_learning_studio(
        total_episodes=args.episodes,
        render_gui=args.render,
        off_road_penalty=args.penalty,
        save_milestone_every=args.save_every,
        resume_checkpoint=args.resume,
        track_seed=args.track_seed,
        initial_speed=args.speed,
    )
