#!/usr/bin/env python3

import gym
import torch
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from robobo_interface import SimulationRobobo, IRobobo
import numpy as np
from collections import deque, namedtuple
from data_files import FIGRURES_DIR

# Define a custom environment for Robobo
class RoboboEnv(gym.Env):
    def __init__(self, rob: IRobobo):
        super(RoboboEnv, self).__init__()
        self.rob = rob
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)  # Continuous action space
        self.observation_space = gym.spaces.Box(low=0, high=1000, shape=(8,), dtype=np.float32)
        self.recent_actions = deque(maxlen=30)  # Memory to track recent actions
        self.total_reward = 0  # Track total reward for the current episode

    def reset(self):
        self.rob.play_simulation()
        self.recent_actions.clear()
        self.total_reward = 0  # Reset total reward
        return self.get_state()

    def step(self, action):
        state = self.get_state()
        left_speed = action[0] * 100  # Scale action to appropriate speed range
        right_speed = action[1] * 100  # Scale action to appropriate speed range

        self.rob.move_blocking(left_speed, right_speed, 100)
        next_state = self.get_state()
        reward = self.get_reward(action, next_state)
        self.total_reward += reward  # Update total reward
        done = self.is_done(next_state, reward)
        self.recent_actions.append(tuple(action))  # Convert action to tuple before appending

        # Debugging logs
        print(f"Action: {action}, Left Speed: {left_speed}, Right Speed: {right_speed}")
        print(f"State: {state}, Next State: {next_state}, Reward: {reward}, Done: {done}")

        return next_state, reward, done, {}

    def get_state(self):
        ir_values = self.rob.read_irs()
        return np.array(ir_values, dtype=np.float32)

    def get_reward(self, action, state):
        if np.any(state > 1000):  # If too close to a wall, set reward to 0
            return 0

        reward = 20 * (action[0] + action[1])  # Reward for moving forward
        reward -= 10 * (1 - (action[0] + action[1]))  # Penalty for moving backward

        # Add penalty for repetitive actions
        if self.recent_actions.count(tuple(action)) > 2:  # Convert action to tuple for comparison
            reward -= 5

        return reward

    def is_done(self, state, reward):
        # End the episode if any IR sensor value is too high (indicating proximity to a wall)
        if np.any(state > 1000):
            self.rob.stop_simulation()
            return True
        # End the episode if the reward is NaN
        if np.isnan(reward):
            self.rob.stop_simulation()
            return True
        return False

    def render(self, mode='human'):
        pass

    def close(self):
        self.rob.stop_simulation()

# Create an instance of SimulationRobobo
robobo_instance = SimulationRobobo()

# Create the environment
env = DummyVecEnv([lambda: RoboboEnv(robobo_instance)])

# Initialize the DDPG agent
model = DDPG('MlpPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Save the model
model.save("ddpg_robobo")

# Load the model
model = DDPG.load("ddpg_robobo")

# Test the trained model
obs = env.reset()
total_rewards = []  # List to track total rewards for each episode
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