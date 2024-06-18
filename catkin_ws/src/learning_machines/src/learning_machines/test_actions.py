#!/usr/bin/env python3

import gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines3.common.callbacks import BaseCallback, StopTrainingOnMaxEpisodes, EvalCallback
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from robobo_interface import SimulationRobobo, IRobobo
import numpy as np
from collections import deque
from data_files import FIGRURES_DIR

class PrintEpisodeCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(PrintEpisodeCallback, self).__init__(verbose)
        self.episode_num = 0

    def _on_step(self) -> bool:
        if self.locals['dones'][0]:
            self.episode_num += 1
            print(f"Episode: {self.episode_num}")
        return True

class SWACallback(BaseCallback):
    def __init__(self, swa_start, swa_freq, verbose=0):
        super(SWACallback, self).__init__(verbose)
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_weights = None
        self.swa_n = 0

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.swa_start and self.num_timesteps % self.swa_freq == 0:
            if self.swa_weights is None:
                self.swa_weights = {k: v.clone() for k, v in self.model.policy.state_dict().items()}
            else:
                for k, v in self.model.policy.state_dict().items():
                    self.swa_weights[k] += v
            self.swa_n += 1
        return True

    def _on_training_end(self) -> None:
        if self.swa_weights is not None:
            for k, v in self.swa_weights.items():
                self.swa_weights[k] = v / self.swa_n
            self.model.policy.load_state_dict(self.swa_weights)

class RoboboEnv(gym.Env):
    def __init__(self, rob: IRobobo):
        super(RoboboEnv, self).__init__()
        self.rob = rob
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)
        self.recent_actions = deque(maxlen=30)
        self.total_reward = 0

    def reset(self):
        self.rob.play_simulation()
        self.recent_actions.clear()
        self.total_reward = 0
        return self.get_state()

    def step(self, action):
        state = self.get_state()
        left_speed = action[0] * 100
        right_speed = action[1] * 100

        self.rob.move_blocking(left_speed, right_speed, 100)
        next_state = self.get_state()
        reward = self.get_reward(action, next_state)
        self.total_reward += reward
        done = self.is_done(next_state, reward)
        self.recent_actions.append(tuple(action))

        return next_state, reward, done, {}

    def get_state(self):
        ir_values = self.rob.read_irs()
        ir_values = np.clip(ir_values, 0, 10000)
        ir_values = ir_values / 10000.0
        return np.array(ir_values, dtype=np.float32)

    def get_reward(self, action, state):
        if np.any(state > 0.7):
            return -20

        reward = 10 * (action[0] + action[1])
        if self.recent_actions.count(tuple(action)) > 5:
            reward -= 10

        if len(self.recent_actions) > 1 and tuple(action) != self.recent_actions[-1]:
            reward += 1

        if len(self.recent_actions) > 0:
            previous_action = np.array(self.recent_actions[-1])
            action_change = np.linalg.norm(action - previous_action)
            reward -= action_change * 5

        return reward

    def is_done(self, state, reward):
        if np.any(state >= 1.0):
            self.rob.stop_simulation()
            return True
        if np.isnan(reward):
            self.rob.stop_simulation()
            return True
        return False

    def render(self, mode='human'):
        pass

    def close(self):
        self.rob.stop_simulation()

robobo_instance = SimulationRobobo()
env = DummyVecEnv([lambda: RoboboEnv(robobo_instance)])
env = VecCheckNan(env, raise_exception=True)

policy_kwargs = dict(
    net_arch=dict(pi=[64, 64], vf=[128, 128]), 
    activation_fn=torch.nn.ReLU,
    normalize_images=True
)

model = PPO('MlpPolicy', env, verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, clip_range=0.1, ent_coef=0.01, policy_kwargs=policy_kwargs)

max_episodes = 1000

print_episode_callback = PrintEpisodeCallback(verbose=1)
stop_training_callback = StopTrainingOnMaxEpisodes(max_episodes=max_episodes, verbose=1)
swa_callback = SWACallback(swa_start=5000, swa_freq=1000, verbose=1)

callback = [print_episode_callback, stop_training_callback, swa_callback]

model.learn(total_timesteps=10000, callback=callback)

save_location = FIGRURES_DIR / "ppo_robobo_100k.zip"
print(f'saving robo to: {save_location}')
model.save(save_location)

obs = env.reset()