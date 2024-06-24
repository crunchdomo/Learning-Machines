import numpy as np
from robobo_interface import SimulationRobobo, IRobobo
import pandas as pd
from data_files import FIGRURES_DIR
import matplotlib.pyplot as plt

import numpy as np
import keras
import random
from collections import deque

class RoboboEnv:
    def __init__(self, rob: IRobobo):
        self.rob = rob
        self.action_space = 2
        self.observation_space = 4
        self.reset()
        self.rewards = []
        
    def reset(self):
        if isinstance(self.rob, SimulationRobobo):
            self.rob.stop_simulation()
            self.rob.play_simulation()
        # self.rob.set_phone_tilt_blocking(100, 100)
        return self.get_state()

    def step(self, action):
        # Simulate movement and update state
        left_speed, right_speed = action

        left_speed = action[0] * 100
        right_speed = action[1] * 100
        self.rob.move_blocking(left_speed, right_speed, 100)

        # Update state based on action (simplified)
        self.state = self.get_state()  # Placeholder for actual sensor readings
        
        # Calculate reward
        reward = (left_speed/2 + right_speed/2)
        
        # Check if done
        done = np.any(self.state >= 0.6)

        self.rewards.append(reward)
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
        
        return self.state, reward, done, {}

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
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.uniform(-1, 1, self.action_size)
        act_values = self.model.predict(state.reshape(1, -1))
        return act_values[0]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape(1, -1))[0])
            target_f = self.model.predict(state.reshape(1, -1))
            target_f[0] = target
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def main(rob):
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

def run_all_actions(rob):
    main(rob)
