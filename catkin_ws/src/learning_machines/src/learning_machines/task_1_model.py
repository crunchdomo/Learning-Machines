#!/usr/bin/env python3
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import namedtuple, deque
import random
import math

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class Model(nn.Module):
    def __init__(self, action_space):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, len(action_space))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
        

    def predict(self, observation) -> tuple:
        # predicts the action and outputs (action, probability of action)
        observation = torch.from_numpy(np.array(observation)).float().unsqueeze(0).to(device)
        probs = self.forward(observation).cpu()
        try:
            eps = np.random.random()
            m = Categorical(probs)
            if eps > 0.85:
                m = Categorical(torch.tensor([.25, .25, .25, .25]))    
        except: # probs are [NaN, NaN, NaN, NaN]
            m = Categorical(torch.tensor([.25, .25, .25, .25]))
        action = m.sample()
        return action.item()


    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)
        

batch_size = 100
gamma = 0.5
tau = 0.9
eps_start = 0.9
eps_end = 0.05
eps_decay = 1000

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(batch_size)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected_state_action_values = (next_state_values * gamma) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def select_action(policy_net, state, steps_done):
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[np.random.randint(4)]], dtype=torch.long)