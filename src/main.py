import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


import argparse
import sys
import os
import gymnasium as gym
from stable_baselines3 import PPO

# Resolve paths relative to project root (one level up from src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

def get_model_path(name):
    # Check in models/ directory first
    p1 = os.path.join(MODELS_DIR, name)
    if os.path.exists(p1) or os.path.exists(p1 + ".zip"):
        return p1
    # Check in current working directory or relative path
    if os.path.exists(name) or os.path.exists(name + ".zip"):
        return name
    return p1


# Function for preprocessing each game frame. 
def preprocess(img):
    img = img[24:84, 17:77] # specific cropping
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
    return img

class ImageEnv(gym.Wrapper):
    def __init__(
        self,
        env,
        skip_frames=4,
        stack_frames=4,
        initial_no_op=50,
        **kwargs
    ):
        super(ImageEnv, self).__init__(env, **kwargs)
        self.initial_no_op = initial_no_op
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames
    
    def reset(self, **kwargs):
        # Reset the original environment.
        s, info = self.env.reset(**kwargs)
        # Do nothing for the next `self.initial_no_op` steps
        for _ in range(self.initial_no_op):
            s, _, _, _, info = self.env.step(0)
        s = preprocess(s)

        # The initial observation is simply k copies of the frame `s`
        self.stacked_state = np.tile(s, (self.stack_frames, 1, 1))  # of size [4, 84, 84]
        return self.stacked_state, info
    
    def step(self, action):
        # We take an action for self.skip_frames steps
        reward = 0
        for _ in range(self.skip_frames):
            s, r, terminated, truncated, info = self.env.step(action)
            reward += r
            if terminated or truncated:
                break

        # Convert a frame to 84 X 84 gray scale one
        s = preprocess(s)

        # Push the current frame `s` at the end of self.stacked_state
        self.stacked_state = np.concatenate((self.stacked_state[1:], s[np.newaxis]), axis=0)

        return self.stacked_state, reward, terminated, truncated, info


class CNNActionValue(nn.Module):
    def __init__(self, state_dim, action_dim): #state_dim = number of frames = 4, action_dim = 5
        super(CNNActionValue, self).__init__()
        self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=8, stride=4)  # [N, 4, 60, 60] -> [N, 16, 14, 14]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)  # [N, 16, 14, 14] -> [N, 32, 6, 6]
        self.in_features = 32 * 6 * 6
        self.fc1 = nn.Linear(self.in_features, 256)
        self.fc2 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view((-1, self.in_features))
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        self.s = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.a = np.zeros((max_size, *action_dim), dtype=np.int64)
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
        ind = np.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.s[ind]),
            torch.FloatTensor(self.a[ind]),
            torch.FloatTensor(self.r[ind]),
            torch.FloatTensor(self.s_prime[ind]),
            torch.FloatTensor(self.terminated[ind]), 
        )
    

class DQN:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=0.0004,
        epsilon=1.0,
        epsilon_min=0.1,
        gamma=0.99,
        batch_size=32,
        warmup_steps=5000,
        buffer_size=int(1e5),
        target_update_interval=10000,
    ):
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_update_interval = target_update_interval

        self.network = CNNActionValue(state_dim[0], action_dim)
        self.target_network = CNNActionValue(state_dim[0], action_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr)

        self.buffer = ReplayBuffer(state_dim, (1, ), buffer_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)
        self.target_network.to(self.device)
        
        self.total_steps = 0
        self.epsilon_decay = (epsilon - epsilon_min) / 6e5
    
    @torch.no_grad()
    def act(self, x, training=True):
        self.network.train(training)
        if training and ((np.random.rand() < self.epsilon) or (self.total_steps < self.warmup_steps)):
            a = np.random.randint(0, self.action_dim)
        else:
            x = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
            q = self.network(x)
            a = torch.argmax(q).item()
        return a
    
    def learn(self):
        s, a, r, s_prime, terminated = map(lambda x: x.to(self.device), self.buffer.sample(self.batch_size))
        
        next_q = self.target_network(s_prime).detach()
        td_target = r + (1. - terminated) * self.gamma * next_q.max(dim=1, keepdim=True).values
        self.optimizer.zero_grad()
        loss = F.mse_loss(self.network(s).gather(1, a.long()), td_target)
        loss.backward()
        self.optimizer.step()
        
        result = {
            'total_steps': self.total_steps,
            'value_loss': loss.item()
        }
        return result
    
    def process(self, transition):
        result = {}
        self.total_steps += 1
        self.buffer.update(*transition)

        if self.total_steps > self.warmup_steps:
            result = self.learn()
            
        if self.total_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        self.epsilon -= self.epsilon_decay
        return result
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate or visualize trained CarRacing-v3 agents (DQN or PPO).")
    parser.add_argument(
        "--agent",
        type=str,
        default="PPO",
        choices=["PPO", "DQN", "ppo", "dqn"],
        help="Agent architecture to run: PPO or DQN (default: PPO)"
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip visual Pygame rendering and only run numerical evaluation."
    )
    parser.add_argument(
        "--eval-first",
        action="store_true",
        help="Run a headless evaluation episode before launching the live visual window."
    )
    parser.add_argument(
        "--allow-offroad",
        action="store_true",
        help="Do not stop simulation if car leaves the asphalt (default is strict: stop when off-track)."
    )
    args = parser.parse_args()
    agent_name = args.agent.upper()

    print(f"\n[CarRacing-v3] Running agent: {agent_name}")
    print("=" * 45)

    try:
        if agent_name == "DQN":
            state_dim = (4, 60, 60)
            action_dim = 5

            # Load the trained parameters of DQN agent
            DQN_agent_test = DQN(state_dim, action_dim)
            dqn_path = get_model_path('true_dqn.pt')
            print(f"Loading DQN weights from {dqn_path}...")
            DQN_agent_test.network.load_state_dict(
                torch.load(dqn_path, map_location=DQN_agent_test.device, weights_only=True)
            )
            DQN_agent_test.network.eval()

            if args.eval_first or args.no_render:
                print("1. Evaluating DQN agent (headless)...")
                env = gym.make("CarRacing-v3", domain_randomize=False, continuous=False, render_mode="rgb_array")
                env = ImageEnv(env)

                (s, _), done, ret = env.reset(), False, 0.0
                step_count = 0
                while not done:
                    a = DQN_agent_test.act(s, training=False)
                    s_prime, r, terminated, truncated, info = env.step(a)
                    s = s_prime
                    ret += r
                    step_count += 1
                    done = terminated or truncated
                env.close()
                print(f"[OK] Headless Evaluation Return: {ret:.2f} (in {step_count} steps)")

            if not args.no_render:
                print("\n🏎️ Launching live visual simulation (render_mode='human')...")
                print("   -> Look for the 'CarRacing-v3' Pygame window on your screen!")
                print("   (Close the game window or press Ctrl+C to stop)")
                sim_env = gym.make("CarRacing-v3", domain_randomize=False, continuous=False, render_mode="human")
                sim_env = ImageEnv(sim_env)

                (s, _), done, sim_ret = sim_env.reset(), False, 0.0
                step_count = 0
                while not done:
                    a = DQN_agent_test.act(s, training=False)
                    s, r, terminated, truncated, info = sim_env.step(a)
                    sim_ret += r
                    step_count += 1
                    if not args.allow_offroad and step_count > 30:
                        car = sim_env.unwrapped.car
                        if car is not None and hasattr(car, 'wheels'):
                            on_road_wheels = sum(1 for w in car.wheels if len(w.tiles) > 0)
                            if on_road_wheels == 0:
                                print(f"\n🚨 OUT OF TRACE! Car went off the track at step {step_count} (0 wheels on asphalt).")
                                print("   Simulation stopped because car left the track boundary.")
                                break
                    if 'lap_complete' in info and info['lap_complete']:
                        print("[LAP] Track completed!")
                    done = terminated or truncated
                sim_env.close()
                print(f"Simulation completed! Steps: {step_count}, Return: {sim_ret:.2f}")

        elif agent_name == "PPO":
            ppo_path = get_model_path("true_ppo_data")
            print(f"Loading PPO model from {ppo_path}...")

            if args.eval_first or args.no_render:
                print("1. Evaluating PPO agent (headless)...")
                env = gym.make("CarRacing-v3", domain_randomize=False, continuous=False, render_mode="rgb_array")
                PPO_agent_test = PPO.load(ppo_path, env=env)

                obs, _ = env.reset()
                ret = 0.0
                step_count = 0
                while True:
                    action, _ = PPO_agent_test.predict(obs, deterministic=True)
                    obs, r, terminated, truncated, info = env.step(int(action))
                    ret += r
                    step_count += 1
                    if terminated or truncated:
                        break
                env.close()
                print(f"[OK] Headless Evaluation Return: {ret:.2f} (in {step_count} steps)")

            if not args.no_render:
                print("\n🏎️ Launching live visual simulation (render_mode='human')...")
                print("   -> Look for the 'CarRacing-v3' Pygame window on your screen!")
                print("   (Close the game window or press Ctrl+C to stop)")
                sim_env = gym.make("CarRacing-v3", domain_randomize=False, continuous=False, render_mode="human")
                PPO_agent_test = PPO.load(ppo_path, env=sim_env)
                obs, _ = sim_env.reset()
                sim_ret = 0.0
                step_count = 0
                while True:
                    action, _ = PPO_agent_test.predict(obs, deterministic=True)
                    obs, r, terminated, truncated, info = sim_env.step(int(action))
                    sim_ret += r
                    step_count += 1
                    if not args.allow_offroad and step_count > 30:
                        car = sim_env.unwrapped.car
                        if car is not None and hasattr(car, 'wheels'):
                            on_road_wheels = sum(1 for w in car.wheels if len(w.tiles) > 0)
                            if on_road_wheels == 0:
                                print(f"\n🚨 OUT OF TRACE! Car went off the track at step {step_count} (0 wheels on asphalt).")
                                print("   Simulation stopped because car left the track boundary.")
                                break
                    if 'lap_complete' in info and info['lap_complete']:
                        print("[LAP] Track completed!")
                    if terminated or truncated:
                        break
                sim_env.close()
                print(f"Simulation completed! Steps: {step_count}, Return: {sim_ret:.2f}")

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
        sys.exit(0)