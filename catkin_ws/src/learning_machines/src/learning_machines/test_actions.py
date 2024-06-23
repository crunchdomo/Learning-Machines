#!/usr/bin/env python3

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from robobo_interface import SimulationRobobo, IRobobo
import numpy as np
import pandas as pd
from data_files import FIGRURES_DIR
from .process_image import process_image as pic_calcs
import matplotlib.pyplot as plt

class RoboboEnv(gym.Env):
    def __init__(self, rob: IRobobo):
        super(RoboboEnv, self).__init__()
        self.rob = rob
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)
        self.total_reward = 0
        self.last_reward = 0
        self.reward = 0
        self.rewards = []
        self.done = False

    def reset(self):
        if isinstance(self.rob, SimulationRobobo):
            self.rob.stop_simulation()
            self.rob.play_simulation()
        # self.rob.set_phone_tilt_blocking(100, 100)
        self.last_reward = 0
        self.done = False
        return self.get_state()

    def step(self, action):
        left_speed = action[0] * 100
        right_speed = action[1] * 100
        self.rob.move_blocking(left_speed, right_speed, 100)

        next_state = self.get_state()

        self.total_reward = self.get_reward(next_state, action)
        self.reward = self.total_reward - self.last_reward
        self.last_reward = self.total_reward

        if self.is_done(next_state):
            self.reward = -5
            self.done = True

        self.rewards.append(self.reward)

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

        return next_state, self.reward, self.done, {}

    def get_state(self):
        # IR values
        ir_values = self.rob.read_irs()
        ir_values = np.clip(ir_values, 0, 10000)
        ir_values = ir_values / 10000.0

        state_values = ir_values#[[7,4,5,6]]
        return np.array(state_values, dtype=np.float32)

    def get_reward(self, state, action):
        return (action[0]/2 + action[1]/2)    
    
    def is_done(self, state):
        IRs = np.array(state)
        if np.any(IRs >= 0.6):
            return True

    def render(self, mode='human'):
        pass

    def close(self):
        self.rob.stop_simulation()

def run_all_actions(rob):

    train = True # Toggle me to switch between training and testing
    save_location = FIGRURES_DIR / "ppo_test.zip"

    if isinstance(rob, SimulationRobobo):
        env = DummyVecEnv([lambda: RoboboEnv(rob)])
        
        if train:
            model = PPO('MlpPolicy', env, verbose=1, 
                        ent_coef=0.01,
                        learning_rate=3e-4,
                        n_steps=2048,
                        batch_size=64,
                        n_epochs=10,
                        gamma=0.99,
                        gae_lambda=0.95,
                        clip_range=0.2)
            
            max_episodes = 1000
            TIMESTEPS = 2048
            env.reset()
            total_timesteps = 0
            episode = 0

            while episode < max_episodes:
                episode += 1
                print(f"ep:{episode}")

                model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
                total_timesteps += TIMESTEPS

                model.save(save_location)
                print(f"Total timesteps: {total_timesteps}")

        else:
            model = PPO.load(save_location, env=env, device='auto')

            model.policy.eval()

            obs = env.reset()
            done = False
            while True:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, _, info = env.step(action)
