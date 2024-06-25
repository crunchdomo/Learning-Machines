#!/usr/bin/env python3

import numpy as np
from robobo_interface import SimulationRobobo, IRobobo
import pandas as pd
from data_files import FIGRURES_DIR
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import cv2

def process_image(rob, image_path, colour='green'):
    # Load the image
    image = cv2.imread(image_path)

    # Resize the image to speed up processing
    scale_percent = 50
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

    if colour == 'green':
        # Define range for green color and create a mask
        lower_green = np.array([40, 100, 100])
        # lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

    if colour == 'red':
        # Define range for red color and create a mask
        # Red wraps around the HSV color space, so we need two ranges
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
            
        # Create masks for both ranges
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            
        # Combine the masks
        mask = cv2.bitwise_or(mask1, mask2)

    height, width = mask.shape
    column_w = width // 3
    
    column_percentages = []
    
    for j in range(3):
        # Extract the column
        column = mask[:, j*column_w:(j+1)*column_w]
        
        # Calculate the percentage of the column filled with the color
        total_pixels = column.size
        colored_pixels = np.count_nonzero(column)
        percentage = colored_pixels / total_pixels
        column_percentages.append(percentage)

    return column_percentages

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Sigmoid()
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
        self.observation_space = 3
        self.reset()
        self.rewards = []
        self.reward = 0
        self.steps = 0
        self.food_consumed = 0
        self.done = False
        
    def reset(self):
        if isinstance(self.rob, SimulationRobobo):
            self.rob.stop_simulation()
            self.rob.play_simulation()
        self.rob.set_phone_tilt_blocking(100, 100)
        self.reward = 0
        self.steps = 0
        self.food_consumed = 0
        self.done = False
        return self.get_state()

    def step(self, action):
        self.steps += 1
        # Simulate movement and update state
        left_speed, right_speed = action

        self.rob.move_blocking(left_speed*50, right_speed*50, 100)

        self.state = self.get_state()

        self.reward = self.get_reward()

        if self.steps > 100:
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
        
        return self.state, self.reward, self.done, {}

    def get_state(self):
        # GCV values
        i = self.rob.get_image_front()
        image = cv2.flip(i, -1)
        # if isinstance(self.rob, SimulationRobobo):
        #     image = cv2.flip(image, 0)
        cv2.imwrite(str(FIGRURES_DIR / "pic.png"), image)
        green_values = process_image(self.rob, str(FIGRURES_DIR / "pic.png"), 'green')
        red_values = process_image(self.rob, str(FIGRURES_DIR / "pic.png"), 'red')

        return np.array(red_values, dtype=np.float32)
    
    def get_reward(self):
        weights = [0.5,1,0.5] 
        
        reward = sum(w * p for w, p in zip(weights, self.state))
        return reward - 1

def run_all_actions(rob):
    env = RoboboEnv(rob)
    state_size = env.observation_space
    print("fl ", state_size)
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


# For Adam
# rob = SimulationRobobo()
# run_all_actions(rob)