"""
car_racing_training.py
======================
Training pipeline and utilities for CarRacing-v3 DQN & PPO agents.
Includes: preprocessing, environment wrappers, DQN implementation,
training loops, evaluation, recording, and plotting.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO

# Project root is one level up from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RECORDINGS_DIR = PROJECT_ROOT / "recorded_races"
MODELS_DIR = PROJECT_ROOT / "models"


def ensure_directories() -> None:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    RECORDINGS_DIR.mkdir(exist_ok=True)


def preprocess_frame(img: np.ndarray) -> np.ndarray:
    """Crop 96x96 RGB observation to 60x60 grayscale normalized float."""
    crop = img[24:84, 17:77]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return gray.astype(np.float32) / 255.0


class ImageEnv(gym.Wrapper):
    """Wraps CarRacing-v3 with frame skipping, stacking, and preprocessing."""
    def __init__(self, env, skip_frames=4, stack_frames=4, initial_no_op=50, **kwargs):
        super().__init__(env)
        self.initial_no_op = initial_no_op
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.initial_no_op):
            obs, _, _, _, info = self.env.step(0)
        state = preprocess_frame(obs)
        self.stacked_state = np.tile(state, (self.stack_frames, 1, 1))
        return self.stacked_state, info

    def step(self, action):
        reward_total = 0.0
        terminated = False
        truncated = False
        info = {}
        for _ in range(self.skip_frames):
            obs, reward, terminated, truncated, info = self.env.step(action)
            reward_total += reward
            if terminated or truncated:
                break
        state = preprocess_frame(obs)
        self.stacked_state = np.concatenate((self.stacked_state[1:], state[np.newaxis]), axis=0)
        return self.stacked_state, reward_total, terminated, truncated, info


class CNNActionValue(nn.Module):
    """CNN that maps stacked frames to Q-values for each action."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.in_features = 32 * 6 * 6
        self.fc1 = nn.Linear(self.in_features, 256)
        self.fc2 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.in_features)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ReplayBuffer:
    """Fixed-size circular buffer for experience replay."""
    def __init__(self, state_dim, max_size=int(1e5)):
        self.s = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.a = np.zeros((max_size,), dtype=np.int64)
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
        self.terminated[self.ptr] = terminated
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.s[ind]),
            torch.LongTensor(self.a[ind]),
            torch.FloatTensor(self.r[ind]),
            torch.FloatTensor(self.s_prime[ind]),
            torch.FloatTensor(self.terminated[ind]),
        )


class DQN:
    """Deep Q-Network agent with experience replay and target network."""
    def __init__(self, state_dim, action_dim, lr=4e-4, epsilon=1.0, epsilon_min=0.1,
                 gamma=0.99, batch_size=32, warmup_steps=5000, buffer_size=int(1e5),
                 target_update_interval=10000):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_update_interval = target_update_interval
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = (epsilon - epsilon_min) / 6e5

        self.network = CNNActionValue(state_dim[0], action_dim)
        self.target_network = CNNActionValue(state_dim[0], action_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr=lr)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)
        self.target_network.to(self.device)

        self.buffer = ReplayBuffer(state_dim, buffer_size)
        self.total_steps = 0

    @torch.no_grad()
    def act(self, x, training=True):
        self.network.train(training)
        if training and ((np.random.rand() < self.epsilon) or (self.total_steps < self.warmup_steps)):
            return int(np.random.randint(0, self.action_dim))
        x = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
        q_values = self.network(x)
        return int(torch.argmax(q_values).item())

    def learn(self):
        s, a, r, s_prime, terminated = map(lambda x: x.to(self.device), self.buffer.sample(self.batch_size))
        next_q = self.target_network(s_prime).detach()
        td_target = r + (1.0 - terminated) * self.gamma * next_q.max(dim=1, keepdim=True).values
        self.optimizer.zero_grad()
        current_q = self.network(s).gather(1, a.view(-1, 1).long())
        loss = F.mse_loss(current_q, td_target)
        loss.backward()
        self.optimizer.step()
        return {"value_loss": float(loss.item())}

    def process(self, transition):
        s, a, r, s_prime, terminated = transition
        self.total_steps += 1
        self.buffer.update(s, a, r, s_prime, float(terminated))

        result = {}
        if self.total_steps > self.warmup_steps:
            result = self.learn()
        if self.total_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        if self.total_steps > self.warmup_steps:
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
        return result

    def save_checkpoint(self, path: str | Path):
        payload = {
            "network_state_dict": self.network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str | Path):
        payload = torch.load(path, map_location=self.device)
        self.network.load_state_dict(payload["network_state_dict"])
        self.target_network.load_state_dict(payload["target_network_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.epsilon = payload.get("epsilon", self.epsilon)
        self.total_steps = payload.get("total_steps", self.total_steps)
        self.network.to(self.device)
        self.target_network.to(self.device)


def load_reference_dqn(path: Path | str | None = None) -> DQN:
    """Load the pre-trained DQN agent from saved weights."""
    env = gym.make("CarRacing-v3", domain_randomize=False, continuous=False, render_mode="rgb_array")
    env = ImageEnv(env)
    state_dim = (4, 60, 60)
    action_dim = env.action_space.n
    dqn = DQN(state_dim, action_dim)
    if path is None:
        path = MODELS_DIR / "true_dqn.pt"
    dqn.network.load_state_dict(torch.load(path, map_location=dqn.device))
    dqn.network.eval()
    return dqn


def load_reference_ppo(path: str | Path | None = None) -> PPO:
    """Load the pre-trained PPO agent from saved weights."""
    if path is None:
        candidate = MODELS_DIR / "true_ppo_data"
        zip_candidate = MODELS_DIR / "true_ppo_data.zip"
        if candidate.exists():
            return PPO.load(str(candidate))
        if zip_candidate.exists():
            return PPO.load(str(zip_candidate))
        raise FileNotFoundError("No PPO checkpoint found. Expected true_ppo_data or true_ppo_data.zip in models/")
    return PPO.load(str(path))


def save_frames_as_mp4(frames: list[np.ndarray], output_path: str | Path, fps: int = 20) -> str:
    """Save a list of RGB frames as an MP4 video file."""
    frames = [np.asarray(frame) for frame in frames]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(output_path), frames, fps=fps)
    return str(output_path)


def evaluate_agent(agent_name: str, record_path: str | Path | None = None, max_steps: int = 1500):
    """Run one episode with a trained agent and optionally record video."""
    env = gym.make("CarRacing-v3", domain_randomize=False, continuous=False, render_mode="rgb_array")
    obs, _ = env.reset()
    total_reward = 0.0
    frames = []
    completed = False
    step_count = 0

    if agent_name == "dqn":
        agent = load_reference_dqn()
        state = obs
        while step_count < max_steps:
            action = agent.act(state, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state
            total_reward += reward
            if record_path is not None:
                frame = env.render()
                frames.append(frame)
            step_count += 1
            completed = bool(info.get("lap_complete", False) or info.get("completed_lap", False))
            if terminated or truncated:
                break
    elif agent_name == "ppo":
        agent = load_reference_ppo()
        while step_count < max_steps:
            action, _ = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(np.array(action))
            total_reward += reward
            if record_path is not None:
                frame = env.render()
                frames.append(frame)
            step_count += 1
            completed = bool(info.get("lap_complete", False) or info.get("completed_lap", False))
            if terminated or truncated:
                break
    else:
        raise ValueError(f"Unsupported agent name: {agent_name}")

    env.close()
    if record_path is not None:
        save_frames_as_mp4(frames, record_path, fps=20)

    return {
        "model": agent_name,
        "reward": float(total_reward),
        "steps": int(step_count),
        "completed": bool(completed),
        "record_path": str(record_path) if record_path is not None else None,
    }


def plot_training_history(history_df: pd.DataFrame, save_path: str | Path | None = None):
    """Plot episode rewards and lengths from training history."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    axes[0].plot(history_df["episode"], history_df["reward"], marker="o", linewidth=1.5)
    axes[0].set_title("Episode Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[1].plot(history_df["episode"], history_df["length"], marker="o", linewidth=1.5)
    axes[1].set_title("Episode Length")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")
    if save_path is not None:
        fig.savefig(save_path, dpi=200)
    return fig


def train_dqn_for_episodes(total_episodes: int = 100, max_steps_per_episode: int = 2000,
                           resume_from_checkpoint: bool = True, save_every: int = 10):
    """Train the DQN agent for a specified number of episodes."""
    ensure_directories()
    env = gym.make("CarRacing-v3", domain_randomize=False, continuous=False, render_mode="rgb_array")
    env = ImageEnv(env)
    state_dim = (4, 60, 60)
    agent = DQN(state_dim, env.action_space.n)

    checkpoint_candidates = sorted(CHECKPOINT_DIR.glob("dqn_*.pt"), key=lambda p: p.name)
    if resume_from_checkpoint and checkpoint_candidates:
        latest_checkpoint = checkpoint_candidates[-1]
        agent.load_checkpoint(latest_checkpoint)
    elif (MODELS_DIR / "true_dqn.pt").exists():
        agent.network.load_state_dict(torch.load(MODELS_DIR / "true_dqn.pt", map_location=agent.device))
        agent.target_network.load_state_dict(agent.network.state_dict())

    history = []

    for episode_idx in range(1, total_episodes + 1):
        state, _ = env.reset()
        done = False
        episode_reward = 0.0
        episode_steps = 0
        while not done and episode_steps < max_steps_per_episode:
            action = agent.act(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.process((state, action, reward, next_state, float(terminated or truncated)))
            state = next_state
            episode_reward += reward
            episode_steps += 1
            done = terminated or truncated

        result = {
            "episode": episode_idx,
            "reward": float(episode_reward),
            "length": int(episode_steps),
            "completed": bool(info.get("lap_complete", False) or info.get("completed_lap", False)),
            "epsilon": float(agent.epsilon),
            "total_steps": int(agent.total_steps),
        }
        history.append(result)
        print(f"Episode {episode_idx}/{total_episodes} | Reward: {episode_reward:.1f} | Steps: {episode_steps}")

        if episode_idx % save_every == 0 or episode_idx == total_episodes:
            checkpoint_path = CHECKPOINT_DIR / f"dqn_episode_{episode_idx:03d}.pt"
            agent.save_checkpoint(checkpoint_path)

    env.close()
    history_df = pd.DataFrame(history)
    history_df.to_csv(PROJECT_ROOT / "episode_metrics.csv", index=False)
    return history_df, agent


if __name__ == "__main__":
    ensure_directories()
    history_df, trained_agent = train_dqn_for_episodes(
        total_episodes=100, max_steps_per_episode=2000,
        resume_from_checkpoint=True, save_every=10
    )
    plot_training_history(history_df, save_path=PROJECT_ROOT / "training_progress.png")
    print("Training complete.")
    print(history_df.tail().to_string(index=False))
