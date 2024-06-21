#!/usr/bin/env python3

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from robobo_interface import SimulationRobobo, IRobobo
import numpy as np
import pandas as pd
from collections import deque
from data_files import FIGRURES_DIR
from .process_image import process_image as pic_calcs
import cv2
import matplotlib.pyplot as plt

class RoboboEnv(gym.Env):
    def __init__(self, rob: IRobobo):
        super(RoboboEnv, self).__init__()
        self.rob = rob
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.total_reward = 0
        self.last_reward = 0
        self.reward = 0
        self.rewards = []
        self.done = False

    def reset(self):
        self.rob.play_simulation()
        self.rob.set_phone_tilt_blocking(100, 100)
        self.last_reward = 0
        self.done = False
        return self.get_state()

    def step(self, action):
        left_speed = action[0] * 100
        right_speed = action[1] * 100
        self.rob.move_blocking(left_speed, right_speed, 500)

        next_state = self.get_state()

        self.total_reward = self.get_reward(next_state)

        self.reward = self.total_reward - self.last_reward

        self.rewards.append(self.reward)

        self.last_reward = self.total_reward

        rewards_series = pd.Series(self.rewards)
        rolling_avg = rewards_series.rolling(window=100).mean()
        
        if len(self.rewards) % 4 == 0:
            plt.figure(figsize=(12, 6))
            plt.plot(self.rewards, label='Reward at each step')
            plt.plot(rolling_avg, label=f'Rolling Average (window size {100})', color='red')
            plt.xlabel('Steps')
            plt.ylabel('Reward')
            plt.legend()
            plt.title('Reward over Time')
            plt.savefig(FIGRURES_DIR / f'training_rewards.png')
            plt.close()

        self.done = self.is_done(next_state, self.reward)

        return next_state, self.reward, self.done, {}

    def get_state(self):
        # CV values
        image = self.rob.get_image_front()
        if isinstance(self.rob, SimulationRobobo):
            image = cv2.flip(image, 0)
        cv2.imwrite(str(FIGRURES_DIR / "pic.png"), image)
        image_values = pic_calcs(str(FIGRURES_DIR / "pic.png"))

        return np.array(image_values, dtype=np.float32)

    def get_reward(self, CV):
        # reward = how close to bottom - how far from middle
        # best score is 1. worst is -0.5
        reward = (CV[0]/2) - abs(CV[1] - 0.5)

        return reward

    def is_done(self, state, reward):
        IRs = np.array(self.rob.read_irs())
        if np.any(IRs >= 1000):
            self.rob.stop_simulation()
            return True
        return False

    def render(self, mode='human'):
        pass

    def close(self):
        self.rob.stop_simulation()


# rob = SimulationRobobo()

def run_all_actions(rob):

    train = True # Toggle me to switch between training and testing
    save_location = FIGRURES_DIR / "ppo_test.zip"

    if isinstance(rob, SimulationRobobo):
        env = DummyVecEnv([lambda: RoboboEnv(rob)])
        if train:
            model = PPO('MlpPolicy', env, verbose=1)

            max_episodes = 1000
            TIMESTEPS = 100

            env.reset()

            episode = 0
            while episode < max_episodes:
                episode += 1
                print(f"ep:{episode}")
                model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)

                model.save(save_location)


        else:
            model = PPO.load(save_location, env=env, device='auto')

            model.policy.eval()

            obs = env.reset()
            done = False
            while True:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, _, info = env.step(action)
