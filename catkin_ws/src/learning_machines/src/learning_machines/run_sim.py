import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import time
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


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class DQNAgent:
    def __init__(self, state_dim, action_dim, memory_capacity, batch_size, gamma, lr):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = ReplayMemory(memory_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
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
                action = self.policy_net(state).argmax().view(1, 1)
                return action
        else:
            action = torch.tensor([[random.randrange(self.action_dim)]], device=device, dtype=torch.long)
            return action

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).long()  # Convert to int64
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        print("State Batch:", state_batch)
        print("Action Batch:", action_batch)
        print("Reward Batch:", reward_batch)
        print("Next State Batch:", next_state_batch)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

def run_simulation(rob: IRobobo, agent: DQNAgent, num_episodes: int):
    for episode in range(num_episodes):
        rob.play_simulation()
        state = get_state(rob)
        start_position = rob.get_position()
        for t in count():
            action = agent.select_action(state)
            left_speed = (action[0].item() * 2 - 1)  # Scale action to wheel speed
            right_speed = (action[0].item() * 2 - 1)   # Scale action to wheel speed
            rob.move(left_speed, right_speed, 100)  # Example action
            next_state = get_state(rob)
            reward = get_reward(rob, start_position)
            done = is_done(rob)

            agent.memory.push(Transition(state, action, next_state, reward))
            state = next_state

            agent.optimize_model()
            if done:
                rob.stop_simulation()
                print("STOPPING CONDITIONSS")
                break
        agent.update_target_network()


def get_state(rob: IRobobo):
    ir_values = rob.read_irs()
    print(ir_values)
    state = torch.tensor([ir_values], device=device, dtype=torch.float)
    return state

def get_reward(rob: IRobobo, start_pos):
    reward = 0
    IRs = rob.read_irs()
    for ir in IRs:
        if ir > 200:
            reward -= 10

    current_pos = rob.get_position()
    
    distance = np.linalg.norm(np.array([current_pos.x, current_pos.y, current_pos.z]) - np.array([start_pos.x, start_pos.y, start_pos.z]))

    reward += (distance*10)

    # Add penalty for falling off the stage
    if is_out_of_bounds(current_pos):
        reward -= 100

    return torch.tensor([reward], device=device)

def is_done(rob: IRobobo):
    current_pos = rob.get_position()
    # Terminate if the robot is out of bounds
    if is_out_of_bounds(current_pos):
        return True
    return False

def is_out_of_bounds(position):
    # Define the boundaries of the stage
    x_min, x_max = -10, 10
    y_min, y_max = -10, 10
    z_min, z_max = 0, 10  # Assuming the stage is at z=0 and has a height of 10 units

    if not (x_min <= position.x <= x_max and y_min <= position.y <= y_max and z_min <= position.z <= z_max):
        return True
    return False

# Initialize the agent
state_dim = 8  # Number of IR sensors
action_dim = 2  # Number of actions (left and right wheel speeds)
agent = DQNAgent(state_dim, action_dim, memory_capacity=10000, batch_size=64, gamma=0.99, lr=1e-3)

# Run the simulation
run_simulation(SimulationRobobo(), agent, num_episodes=1000)


def test_run():

    # Initialize the agent
    state_dim = 8  # Number of IR sensors
    action_dim = 2  # Number of actions (left and right wheel speeds)
    agent = DQNAgent(state_dim, action_dim, memory_capacity=10000, batch_size=64, gamma=0.99, lr=1e-3)

    # Run the simulation
    run_simulation(SimulationRobobo(), agent, num_episodes=1000)


# NEEDS TO DO: 
# 1) Get model saved 
# 2) Manage Errors
