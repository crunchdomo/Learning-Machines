#!/usr/bin/env python3

import gym
import torch
from stable_baselines3 import DQN
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
        self.action_space = gym.spaces.Discrete(4)  # Forward, Backward, Turn Left, Turn Right
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
        if action == 0:
            left_speed = 40
            right_speed = 40
            movement = 'forward'
        elif action == 1:
            left_speed = -40
            right_speed = -40
            movement = 'backward'
        elif action == 2:
            left_speed = 40
            right_speed = -40
            movement = 'turn_left'
        elif action == 3:
            left_speed = -40
            right_speed = 40
            movement = 'turn_right'

        self.rob.move_blocking(left_speed, right_speed, 100)
        next_state = self.get_state()
        reward = self.get_reward(action, next_state, movement)
        self.total_reward += reward  # Update total reward
        done = self.is_done(next_state)
        self.recent_actions.append(action)
        return next_state, reward, done, {}

    def get_state(self):
        ir_values = self.rob.read_irs()
        return np.array(ir_values, dtype=np.float32)

    def get_reward(self, action, state, movement):
        if np.any(state > 1000):  # If too close to a wall, set reward to 0
            return 0

        if movement == 'forward':
            reward = 20  # Stronger reward for moving forward
        elif movement == 'backward':
            reward = -50  # Stronger penalty for moving backward
        elif movement in ['turn_left', 'turn_right']:
            reward = 5  # Small reward for turning to encourage exploration

        # Add penalty for repetitive actions
        if self.recent_actions.count(action) > 2:
            reward -= 5

        return reward

    def is_done(self, state):
        # End the episode if any IR sensor value is too high (indicating proximity to a wall)
        if np.any(state > 100):
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

# Initialize the DQN agent
model = DQN('MlpPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Save the model
model.save("dqn_robobo")

# Load the model
model = DQN.load("dqn_robobo")

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
