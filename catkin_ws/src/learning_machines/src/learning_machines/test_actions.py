#!/usr/bin/env python3

import gym
import torch
from stable_baselines3 import TD3  
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines3.common.noise import NormalActionNoise
from robobo_interface import SimulationRobobo, IRobobo
import numpy as np
from collections import deque, namedtuple
from data_files import FIGRURES_DIR

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

        # Debugging logs
        print(f"Action: {action}, Left Speed: {left_speed}, Right Speed: {right_speed}")
        print(f"State: {state}, Next State: {next_state}, Reward: {reward}, Done: {done}")

        return next_state, reward, done, {}

    def get_state(self):
        ir_values = self.rob.read_irs()
        ir_values = np.clip(ir_values, 0, 10000)  
        ir_values = ir_values / 10000.0  
        if np.any(np.isinf(ir_values)) or np.any(np.isnan(ir_values)):
            print("Invalid IR sensor values detected:", ir_values)
        return np.array(ir_values, dtype=np.float32)

    def get_reward(self, action, state):
        if np.any(state > 1.0):  
            return 0

        reward = 20 * (action[0] + action[1])
        reward -= 10 * (1 - (action[0] + action[1])) 


        if self.recent_actions.count(tuple(action)) > 2: 
            reward -= 10 

        if len(self.recent_actions) > 1 and tuple(action) != self.recent_actions[-1]:
            reward += 1

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


n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3('MlpPolicy', env, verbose=1)


model.learn(total_timesteps=10000)


model.save("td3_robobo")

model = TD3.load("td3_robobo")

obs = env.reset()
total_rewards = [] 
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(reward)
    total_rewards.append(reward)  # Track the reward
    if done:
        break
    env.render()

# Print the total rewards for each episode
print("Total rewards for each episode:", total_rewards)