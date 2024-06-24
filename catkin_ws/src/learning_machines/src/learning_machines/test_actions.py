import numpy as np
from robobo_interface import SimulationRobobo, IRobobo
import pandas as pd
from data_files import FIGRURES_DIR
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class RoboboEnv:
    def __init__(self, rob: IRobobo):
        self.rob = rob
        self.action_space = 2
        self.observation_space = 4
        self.reset()
        self.rewards = []
        self.reward = 0
        
    def reset(self):
        if isinstance(self.rob, SimulationRobobo):
            self.rob.stop_simulation()
            self.rob.play_simulation()
        # self.rob.set_phone_tilt_blocking(100, 100)
        self.reward = 0
        return self.get_state()

    def step(self, action):
        # Simulate movement and update state
        left_speed, right_speed = action

        self.rob.move_blocking(left_speed*100, right_speed*100, 100)

        # Update state based on action (simplified)
        self.state = self.get_state()

        # Calculate reward
        forward_movement = (left_speed + right_speed) / 2
        collision_penalty = -np.mean(self.state)  # Higher IR values indicate closer obstacles
        stop_penalty = -1 if forward_movement < 0.1 else 0  # Penalize if the robot is almost stopping

        self.reward = forward_movement + collision_penalty + stop_penalty
        
        # Check if done
        done = np.any(self.state >= 0.6)
        if done:
            self.reward = -1

        self.reward = np.clip(self.reward, -1, 1)

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
        
        return self.state, self.reward, done, {}

    def get_state(self):
        # IR values
        ir_values = self.rob.read_irs()
        ir_values = np.clip(ir_values, 0, 10000)
        ir_values = ir_values / 10000.0

        state_values = ir_values[[7,4,5,6]]
        return np.array(state_values, dtype=np.float32)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.uniform(0, 1, self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act_values = self.model(state)
        return np.clip(act_values.cpu().numpy()[0], 0, 1)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            action = torch.FloatTensor(action).unsqueeze(0).to(self.device)
            
            target = reward
            if not done:
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(self.model(next_state)).item()
            
            current_q_values = self.model(state)
            target_q_values = current_q_values.clone()
            target_q_values[:, :] = target
            
            self.optimizer.zero_grad()
            loss = self.criterion(current_q_values, target_q_values)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def run_all_actions(rob):
    env = RoboboEnv(rob)
    state_size = env.observation_space
    action_size = env.action_space
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    episodes = 1000

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")
