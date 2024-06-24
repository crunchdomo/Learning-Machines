import numpy as np
from pyqlearning.functionapproximator import FunctionApproximator
from pyqlearning.q_learning import QLearning
from robobo_interface import SimulationRobobo, IRobobo
import pandas as pd
from data_files import FIGRURES_DIR
import matplotlib.pyplot as plt

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

class RoboboDQN(FunctionApproximator):
    def __init__(self, input_dim, output_dim):
        super(RoboboDQN, self).__init__(input_dim, output_dim)
        # Initialize your neural network here
        # For simplicity, we'll use a basic linear model
        self.weights = np.random.rand(input_dim, output_dim)

    def learn(self, x, y):
        # Implement learning algorithm (e.g., gradient descent)
        learning_rate = 0.01
        prediction = self.inference(x)
        error = y - prediction
        self.weights += learning_rate * np.outer(x, error)

    def inference(self, x):
        return np.dot(x, self.weights)

def main(rob):
    env = RoboboEnv(rob)
    dqn = RoboboDQN(env.observation_space, env.action_space)
    
    q_learning = QLearning(
        gamma=0.99,
        alpha=0.1,
        epsilon=0.1,
        function_approximator=dqn,
        state_dim=env.observation_space,
        action_dim=env.action_space
    )

    episodes = 1000
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = q_learning.select_action(state)
            next_state, reward, done, _ = env.step(action)
            q_learning.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

def run_all_actions(rob):
    main(rob)
