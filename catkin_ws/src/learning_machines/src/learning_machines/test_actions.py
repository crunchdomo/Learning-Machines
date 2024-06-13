#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
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

class RobotControllerNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RobotControllerNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Output range -1 to 1
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            np.array([s.numpy() if isinstance(s, torch.Tensor) else s for s in state]),
            np.array([a.numpy() if isinstance(a, torch.Tensor) else a for a in action]),
            np.array([r.numpy() if isinstance(r, torch.Tensor) else r for r in reward]),
            np.array([ns.numpy() if isinstance(ns, torch.Tensor) else ns for ns in next_state]),
            np.array([d.numpy() if isinstance(d, torch.Tensor) else d for d in done])
        )

    def __len__(self):
        return len(self.buffer)

def train_model(model, optimizer, replay_buffer, batch_size, gamma):
    if len(replay_buffer) < batch_size:
        return

    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    state = torch.FloatTensor(state)
    action = torch.FloatTensor(action).unsqueeze(1)
    reward = torch.FloatTensor(reward)
    next_state = torch.FloatTensor(next_state)
    done = torch.FloatTensor(done)

    q_values = model(state)
    next_q_values = model(next_state)
    q_value = q_values.gather(1, action.long()).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    # Ensure both q_value and expected_q_value have the same shape
    expected_q_value = expected_q_value.unsqueeze(1)

    loss = F.mse_loss(q_value, expected_q_value.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def get_state(rob):
    ir_values = rob.read_irs()
    return torch.tensor([ir_values], device=device, dtype=torch.float)

def reward_function(p1, p2, movement, next_state):
    reward = 0
    if movement == 'forward':
        reward += 10
    elif movement == 'turn':
        reward += 1
    else:
        reward += -10

    distance = np.linalg.norm(np.array([p2.x, p2.y, p2.z]) - np.array([p1.x, p1.y, p1.z]))
    reward += (distance * 10)
    return reward

def run_training(rob):
    input_dim = 8
    output_dim = 1
    model = RobotControllerNN(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    replay_buffer = ReplayBuffer(10000)
    model_path = FIGRURES_DIR / 'best.model'
    batch_size = 64
    gamma = 0.99
    num_episodes = 10

    for episode in range(num_episodes):
        rob.play_simulation()
        state = get_state(rob)
        done = False
        movements = 0

        while movements < 10:
            movements += 1
            p1 = rob.get_position()
            action = model(torch.FloatTensor(state)).detach().numpy()
            speed = action[0]  # Single action value

            # Define movement based on single action value
            if speed >= 0.5:
                left_speed = 100
                right_speed = 100
                movement = 'forward'
            elif 0 <= speed < 0.5:
                left_speed = 100
                right_speed = -100
                movement = 'turn'
            elif -0.5 <= speed < 0:
                left_speed = -100
                right_speed = 100
                movement = 'turn'
            else:
                left_speed = -100
                right_speed = -100
                movement = 'back'

            rob.move_blocking(left_speed, right_speed, 100)
            p2 = rob.get_position()
            next_state = get_state(rob)
            reward = reward_function(p1, p2, movement, next_state)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            print(f"Episode: {episode}, Movements: {movements}, Reward: {reward}")
            print(action)

            train_model(model, optimizer, replay_buffer, batch_size, gamma)

        rob.stop_simulation()

        if episode % 100 == 0:
            print(f"Episode {episode} completed")
            torch.save(model.state_dict(), model_path)
            print("Model saved successfully")

    torch.save(model.state_dict(), model_path)
    print("Model saved successfully")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pass the instance to the function
# rob = SimulationRobobo()
def run_all_actions(rob):
    run_training(rob)
    print('done')
