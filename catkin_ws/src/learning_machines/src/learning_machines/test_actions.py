#!/usr/bin/env python3

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
import cv2

def process_image(image_path, colour='green'):
    """
    Parameters
    ----------
    image_path : str
        The path to the image file.
    Returns
    -------
    tuple
        A tuple containing two float values:
        - normalized_distance_bottom: Closeness to bottom of the image. Returns 1 at bottom, 0 at top.
        - normalized_distance_side: Closeness to right of the image. Returns 1 at the right, 0 at the left.
        - Returns 0 in both cases if no box is found.
    """
    # Load the image
    image = cv2.imread(image_path)

    # Resize the image to speed up processing
    scale_percent = 50  # percent of original size
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
        if isinstance(rob, SimulationRobobo):
            # Define range for blue color and create a mask
            lower_blue = np.array([110, 50, 50])
            upper_blue = np.array([130, 255, 255])
            
            # Create mask for blue color
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
        else:
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

    # Display the mask for debugging
    # cv2.imshow('Mask', mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get image dimensions
    image_height, image_width = resized_image.shape[:2]

    closest_contour = None
    min_distance_from_bottom = float('inf')

    for contour in contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        box_bottom = y + h

        # Calculate the distance to the bottom of the image
        distance_from_bottom = image_height - box_bottom

        if distance_from_bottom < min_distance_from_bottom:
            min_distance_from_bottom = distance_from_bottom
            closest_contour = contour

    if closest_contour is not None:
        # Get the bounding box of the closest contour
        x, y, w, h = cv2.boundingRect(closest_contour)
        box_bottom = y + h
        box_center_x = x + w / 2

        # Calculate the distance from the bottom of the image
        distance_from_bottom = image_height - box_bottom

        # Normalize the distance to a value between 0 and 1
        normalized_distance_bottom = 1 - (distance_from_bottom / image_height)

        # Normalize the horizontal position to a value between 0 and 1
        normalized_distance_right = box_center_x / image_width

        return normalized_distance_bottom, normalized_distance_right
    else:
        return 0, 0

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
        self.observation_space = 8
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
        self.rob.set_phone_tilt_blocking(120, 100)
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

        # Update state based on action (simplified)
        self.state = self.get_state()

        # self.reward = self.state[1] - 1

        # done = self.state[1] == 1
        # if done:
        #     self.reward = 2

        # IR = 0
        # if self.state[0]> 0.1 and self.state[1] > 0.9:
        #     IR = 0.5
        # if self.state[0] > 0.9:
        #     print[self.state[0]]
        # # CV stuff
        # self.reward = 1-self.state[1] + IR

        # if self.state[1] == 1:
        #     self.reward += self.state[3]

        # IR values
        ir_values = self.rob.read_irs()
        ir_values = np.clip(ir_values, 0, 10000)
        ir_values = ir_values / 10000.0

        if self.food_consumed < self.rob.nr_food_collected():
            self.reward += 0.2 * (self.rob.nr_food_collected() - self.food_consumed) * (0 if left_speed < 0 and right_speed < 0 else 1)
            self.food_consumed = self.rob.nr_food_collected()

        # Check if done
        # done = np.any(ir_values >= 0.8)
        # if done:
        #     self.reward = -1
        if self.steps > 100:
            self.done = True

        # self.reward = np.clip(self.reward, -1, 1)

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
        image = self.rob.get_image_front()
        if isinstance(self.rob, SimulationRobobo):
            image = cv2.flip(image, 0)
        cv2.imwrite(str(FIGRURES_DIR / "pic.png"), image)
        green_values = process_image(str(FIGRURES_DIR / "pic.png"), 'green')
        red_values = process_image(str(FIGRURES_DIR / "pic.png"), 'red')

        # IR values
        ir_values = self.rob.read_irs()
        ir_values = np.clip(ir_values, 0, 10000)
        ir_values = ir_values / 10000.0

        # state_values = ir_values[[7,4,5,6]]
        # state_values = image_values
        # state_values = np.concatenate((ir_values[[7,4,5,6]], green_values))
        state_values = np.concatenate((ir_values[[7,4,5,6]], green_values, red_values))
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


# For Adam
rob = SimulationRobobo()
run_all_actions(rob)