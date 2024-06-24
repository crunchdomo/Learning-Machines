import numpy as np
from robobo_interface import SimulationRobobo, IRobobo
import pandas as pd
from data_files import FIGRURES_DIR
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torch.distributions import Normal
import random

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(action_size))

    def forward(self, state):
        value = self.critic(state)
        mu = self.actor(state)
        std = self.log_std.exp()
        dist = Normal(mu, std)
        return dist, value

class PPOAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        self.clip_param = 0.2
        self.ppo_epochs = 10
        self.batch_size = 32

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        dist, _ = self.model(state)
        action = dist.sample()
        return action.cpu().numpy()[0]

    def update(self, states, actions, rewards, next_states, dones):
        # Convert lists of numpy arrays to single numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        # Convert numpy arrays to torch tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute advantages
        with torch.no_grad():
            _, next_values = self.model(next_states)
            advantages = rewards + (1 - dones) * 0.99 * next_values.squeeze() - self.model(states)[1].squeeze()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.ppo_epochs):
            # Compute actor and critic losses
            dist, values = self.model(states)
            log_probs = dist.log_prob(actions).sum(1)
            entropy = dist.entropy().mean()

            ratio = torch.exp(log_probs - log_probs.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = nn.MSELoss()(values.squeeze(), rewards + 0.99 * next_values.squeeze() * (1 - dones))

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            # Update model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

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
        collision_penalty = -np.max(self.state)  # Higher IR values indicate closer obstacles
        # stop_penalty = -1 if forward_movement < 0.1 else 0  # Penalize if the robot is almost stopping

        self.reward = -1 + forward_movement + collision_penalty# + stop_penalty
        
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
            # plt.plot(self.rewards, label='Reward at each step')
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

def run_all_actions(rob):
    env = RoboboEnv(rob)
    state_size = env.observation_space
    action_size = env.action_space
    agent = PPOAgent(state_size, action_size)
    
    episodes = 1000
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        states, actions, rewards, next_states, dones = [], [], [], [], []

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

            state = next_state
            total_reward += reward

            if len(states) >= agent.batch_size:
                agent.update(states, actions, rewards, next_states, dones)
                states, actions, rewards, next_states, dones = [], [], [], [], []

        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")