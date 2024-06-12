import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
from itertools import count
import random
import numpy as np
import os

from data_files import FIGRURES_DIR


# Define the DQN model
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

# Define the ReplayMemory class
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Define the DQNAgent class
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
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if np.random.random() > self.epsilon:
            with torch.no_grad():
                action = self.policy_net(state).argmax().view(1, 1)
            return action
        else:
            return torch.tensor([[random.randrange(self.action_dim)]], device=device, dtype=torch.long)

        
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Assuming batch is a named tuple with fields: state, action, reward, next_state
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).long()#.view(-1, 2).long()
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        # Debug prints
        print("state_batch shape:", state_batch.shape)
        print("action_batch shape:", action_batch.shape)
        print("reward_batch shape:", reward_batch.shape)
        print("next_state_batch shape:", next_state_batch.shape)
        print("action_batch contents:", action_batch)

        # Continue with the rest of the optimization process
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Define helper functions
def get_state(rob):
    ir_values = rob.read_irs()
    state = torch.tensor([ir_values], device=device, dtype=torch.float)
    return state

def get_reward(rob, start_pos, movement):
    reward = 0
    IRs = rob.read_irs()
    for ir in IRs:
        if ir > 1000:
            reward -= 100
    if movement == 'forward':
        reward += 10
    elif movement == 'turn':
        reward += 5
    else:
        reward += -20
    current_pos = rob.get_position()
    distance = np.linalg.norm(np.array([current_pos.x, current_pos.y, current_pos.z]) - np.array([start_pos.x, start_pos.y, start_pos.z]))
    reward += (distance * 10)
    return torch.tensor([reward], device=device)

def run_training_simulation(rob, agent, num_episodes):
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
            action = agent.select_action(state)
            # action = action.view(-1)
            left_speed = action[0].item()
            right_speed = action[0].item()

            rob.move_blocking(left_speed, right_speed, 100)
            next_state = get_state(rob)

            # figure out movement direction based on wheel values
            if left_speed > 0 and right_speed > 0:
                movement = 'forward'
            elif left_speed < 0 and right_speed < 0:
                movement = 'back'
            else:
                movement = 'turn'

            reward = get_reward(rob, start_position, movement)
            total_reward += reward.item()

            agent.memory.push(Transition(state, action, next_state, reward))
            state = next_state

            agent.optimize_model()

            if t > 10:
                rob.stop_simulation()
                break

        if total_reward > best_reward:
            best_reward = total_reward
            #torch.save(agent.policy_net.state_dict(), model_path)
            print(f"Saved best model with reward: {best_reward}")

        agent.update_target_network()

# Initialize the agent and run the simulation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
state_dim = 8
action_dim = 2
agent = DQNAgent(state_dim, action_dim, memory_capacity=10000, batch_size=64, gamma=0.99, lr=1e-3)

# ADAM PLEASE MAINTAIN THIS STRUCTURE TO AVOID AN ERROR. JUST CHANGE THE FUNCTION CALLED IN RUN_ALL_ACTIONS AS REQUIRED
def run_all_actions(rob):
    run_training_simulation(rob, agent, num_episodes=100)
