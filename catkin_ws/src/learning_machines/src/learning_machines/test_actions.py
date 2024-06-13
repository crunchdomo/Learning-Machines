#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import time
import os
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import namedtuple
from itertools import count
import numpy as np

from data_files import FIGRURES_DIR
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)

# Define the neural network model
class SpeedPredictor(nn.Module):
    def __init__(self, input_dim):
        super(SpeedPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)  # Output two values: left_speed and right_speed

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Define the ReplayMemory class
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Define the Agent class
class SpeedAgent:
    def __init__(self, state_dim, memory_capacity, batch_size, gamma, lr):
        self.state_dim = state_dim
        self.memory = ReplayMemory(memory_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.policy_net = SpeedPredictor(state_dim).to(device)
        self.target_net = SpeedPredictor(state_dim).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.steps_done = 0
        self.epsilon = 0.9
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def select_action(self, state):
        sample = random.random()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if sample > self.epsilon:
            with torch.no_grad():
                speeds = self.policy_net(state)
            return speeds
        else:
            speeds = torch.tensor([[random.uniform(-100, 100), random.uniform(-100, 100)]], device=device, dtype=torch.float)
            return speeds

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        speeds_batch = torch.cat(batch.speeds)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        state_speeds_values = self.policy_net(state_batch)

        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_state_speeds_values = (next_state_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(state_speeds_values, expected_state_speeds_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Define helper functions
def get_state(rob):
    ir_values = rob.read_irs()
    state = torch.tensor([ir_values], device=device, dtype=torch.float)
    return state

def get_reward(rob, start_pos, movement, speeds):
    reward = 0
    IRs = rob.read_irs()
    
    # Adjust IR penalties
    for ir in IRs:
        if ir > 1000:
            reward -= 10  

    # Adjust movement rewards
    if movement == 'forward':
        reward += 1000
    elif movement == 'turn':
        reward += 0
    else:
        reward += -10 

    # Penalize zero speeds
    left_speed, right_speed = speeds[0, 0].item(), speeds[0, 1].item()
    if left_speed < 0 and right_speed < 0:
        reward -= 20  # Penalty for staying still

    # Reward continuous movement
    if left_speed != 0 or right_speed != 0:
        reward += 5  # Small reward for any movement

    # Calculate distance reward
    current_pos = rob.get_position()
    distance = np.linalg.norm(np.array([current_pos.x, current_pos.y, current_pos.z]) - np.array([start_pos.x, start_pos.y, start_pos.z]))
    reward += (distance * 10)  # Ensure this scaling is appropriate

    # Penalize unnecessary movements
    if movement == 'back':
        reward -= 100

    # Normalize rewards
    reward = reward / 100.0

    return torch.tensor([reward], device=device)

def run_training_simulation(rob, agent, num_episodes, max_steps_per_episode=100):
    best_reward = -float('inf')
    model_path = FIGRURES_DIR / 'best.model'
    if os.path.exists(model_path):
        agent.policy_net.load_state_dict(torch.load(model_path))
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print("Loaded saved model.")

    for episode in range(num_episodes):
        print(f'Starting episode {episode}')
        rob.play_simulation()
        state = get_state(rob)
        start_position = rob.get_position()
        total_reward = 0

        for t in count():
            speeds = agent.select_action(state)
            left_speed, right_speed = speeds[0, 0].item(), speeds[0, 1].item()
            rob.move_blocking(left_speed, right_speed, 100)
            next_state = get_state(rob)

            if left_speed > 0 and right_speed > 0:
                movement = 'forward'
            elif left_speed < 0 and right_speed < 0:
                movement = 'back'
            else:
                movement = 'turn'

            reward = get_reward(rob, start_position, movement, speeds)
            total_reward += reward.item()

            agent.memory.push(Transition(state, speeds, next_state, reward))
            state = next_state

            agent.optimize_model()

            # Log the reward and state
            print(f"Step: {t}, Reward: {reward.item()}, Total Reward: {total_reward}")

            if t >= max_steps_per_episode:
                rob.stop_simulation()
                break

        agent.update_target_network()
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(agent.policy_net.state_dict(), model_path)
            print(f"Saved best model with reward: {best_reward}")

# Initialize the agent and run the simulation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'speeds', 'next_state', 'reward'))
state_dim = 8

agent = SpeedAgent(state_dim, memory_capacity=10000, batch_size=64, gamma=0.99, lr=1e-3)

def run_all_actions(rob):
    run_training_simulation(rob, agent, num_episodes=1000)

# Create an instance of SimulationRobobo
robobo_instance = SimulationRobobo()

# Pass the instance to the function
run_all_actions(robobo_instance)