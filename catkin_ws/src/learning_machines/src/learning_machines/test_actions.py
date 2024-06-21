#!/usr/bin/env python3

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines3.common.callbacks import BaseCallback, StopTrainingOnMaxEpisodes
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
        self.window = 10
        self.rewards = []
        self.current_step = 0
        self.max_steps = 100
        self.last_reward = 0
        self.reward = 0
        self.done = False

    def reset(self):
        self.rob.play_simulation()
        self.rob.set_phone_tilt_blocking(100, 100)
        self.last_reward = 0
        self.current_step = 0
        self.done = False
        return self.get_state()

    def step(self, action):
        self.current_step += 1
        old_state = self.get_state() # get the CV data for 1st state

        left_speed = action[0] * 100
        right_speed = action[1] * 100
        self.rob.move_blocking(left_speed, right_speed, 100)

        next_state = self.get_state() # get the CV data for 2nd state

        self.total_reward = self.get_reward(action, old_state, next_state)

        self.reward = self.total_reward - self.last_reward

        self.last_reward = self.total_reward

        self.done = self.is_done(next_state, self.reward)

        # Reward plotting stuff
        self.rewards.append(self.reward)
        rewards_series = pd.Series(self.rewards)
        rolling_avg = rewards_series.rolling(window=self.window).mean()
        if self.current_step % 4 == 0:
            plt.figure(figsize=(12, 6))
            plt.plot(self.rewards, label='Reward at each step')
            plt.plot(rolling_avg, label=f'Rolling Average (window size {self.window})', color='red')
            plt.xlabel('Steps')
            plt.ylabel('Reward')
            plt.legend()
            plt.title('Reward over Time')
            plt.savefig(FIGRURES_DIR / f'training_rewards.png')
            plt.close()

        return next_state, self.reward, self.done, {}

    def get_state(self):
        # # IR values
        # ir_values = self.rob.read_irs()
        # ir_values = np.clip(ir_values, 0, 10000)
        # ir_values = ir_values / 10000.0

        # GCV values
        image = self.rob.get_image_front()
        if isinstance(self.rob, SimulationRobobo):
            image = cv2.flip(image, 0)
        cv2.imwrite(str(FIGRURES_DIR / "pic.png"), image)
        image_values = pic_calcs(str(FIGRURES_DIR / "pic.png"))

        # state_values = np.concatenate((ir_values[[7,4,5,6]], image_values))
        state_values = image_values
        return np.array(state_values, dtype=np.float32)

    def get_reward(self, action, old_CV, new_CV):
        reward = 0
        '''
        If rob moves closer to block, reward
        If block reaches bottom of cam and stays there, fine.
        If block moves away, punish
        '''
        # calc for bottom dist
        if old_CV[0] < new_CV[0]:
            reward += new_CV[0] - old_CV[0] * 10
        elif new_CV[0] == 1 and old_CV[0] == 1:
            reward += 1
        elif old_CV[0] > new_CV[0]:
            reward += (new_CV[0] - old_CV[0]) * 50

        # reward += new_CV[0]

        '''
        Reward if block gets closer to middle
        Punish if it gets further from middle
        '''
        # old_distance = abs(old_CV[1] - 0.5)
        # new_distance = abs(new_CV[1] - 0.5)

        # # reward += (old_distance - new_distance) * 10  
        # reward += new_distance

        # if (action[0] + action[1]) < 0:
        #     reward += (action[0] + action[1])
    
        # Checks if x values are BIG. 
        # goal is to not punish for touching food but still account for hitting walls
        # if np.sum(IR > 0.8) >= 1:
        #     reward -= 50

        return reward

    def is_done(self, state, reward):
        if np.isnan(reward)  or self.current_step >= self.max_steps:
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

save_location = FIGRURES_DIR / "ppo_robobo_best.zip"

train = True # Toggle me to switch between training and testing
if train:
    model = PPO('MlpPolicy', env, verbose=1)

    max_episodes = 1000
    TIMESTEPS = 50

    iters = 0
    while True:
        iters += 1
        print(f"ep:{iters}")
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)

        print(f'saving robo to: {save_location}')
        model.save(save_location)


else:
    model = PPO.load(save_location, env=env, device='auto')

    model.policy.eval()

    obs = env.reset()
    done = False
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, _, info = env.step(action)

def run_all_actions(rob):
    print("Finished!!")
