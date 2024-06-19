#!/usr/bin/env python3

import gym
import torch
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines3.common.callbacks import BaseCallback, StopTrainingOnMaxEpisodes, EvalCallback
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from robobo_interface import SimulationRobobo, IRobobo
import numpy as np
from collections import deque
from data_files import FIGRURES_DIR
from .process_image import process_image as pic_calcs
import cv2

class PrintEpisodeCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(PrintEpisodeCallback, self).__init__(verbose)
        self.episode_num = 0

    def _on_step(self) -> bool:
        if self.locals['dones'][0]:
            self.episode_num += 1
            print(f"Episode: {self.episode_num}")
        return True

class SWACallback(BaseCallback):
    def __init__(self, swa_start, swa_freq, verbose=0):
        super(SWACallback, self).__init__(verbose)
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_weights = None
        self.swa_n = 0

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.swa_start and self.num_timesteps % self.swa_freq == 0:
            if self.swa_weights is None:
                self.swa_weights = {k: v.clone() for k, v in self.model.policy.state_dict().items()}
            else:
                for k, v in self.model.policy.state_dict().items():
                    self.swa_weights[k] += v
            self.swa_n += 1
        return True

    def _on_training_end(self) -> None:
        if self.swa_weights is not None:
            for k, v in self.swa_weights.items():
                self.swa_weights[k] = v / self.swa_n
            self.model.policy.load_state_dict(self.swa_weights)

class SaveBestModelCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(SaveBestModelCallback, self).__init__(verbose)
        self.best_mean_reward = -np.inf
        self.episode_rewards = []

    def _on_step(self) -> bool:
        if self.locals['dones'][0]:
            episode_reward = sum(self.locals['rewards'])
            self.episode_rewards.append(episode_reward)
            mean_reward = np.mean(self.episode_rewards[-100:])  # Mean reward over the last 100 episodes
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                save_location = FIGRURES_DIR / f"ppo_robobo_best.zip"
                self.model.save(save_location)
                print(f"New best model saved with mean reward: {mean_reward}")
        return True

class RoboboEnv(gym.Env):
    def __init__(self, rob: IRobobo):
        super(RoboboEnv, self).__init__()
        self.rob = rob
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)
        self.recent_actions = deque(maxlen=30)
        self.total_reward = 0
        self.current_step = 0
        self.max_steps = 100

    def reset(self):
        self.rob.play_simulation()
        self.rob.set_phone_tilt_blocking(100, 100)
        self.recent_actions.clear()
        self.total_reward = 0
        self.current_step = 0
        return self.get_state()

    def step(self, action):
        self.current_step += 1
        old_state = self.get_state() # get the CV data for 1st state
        left_speed = action[0] * 100
        right_speed = action[1] * 100

        self.rob.move_blocking(left_speed, right_speed, 100)
        next_state = self.get_state() # get the CV data for 2nd state
        reward = self.get_reward(action, old_state, next_state)
        self.total_reward += reward
        done = self.is_done(next_state[:4], reward) or self.current_step >= self.max_steps
        if self.is_done(next_state[:4], reward):# or self.current_step >= self.max_steps:
            done = True
            self.rob.stop_simulation()
        self.recent_actions.append(tuple(action))

        return next_state, reward, done, {}

    def get_state(self):
        # IR values
        ir_values = self.rob.read_irs()
        ir_values = np.clip(ir_values, 0, 10000)
        ir_values = ir_values / 10000.0

        # GCV values
        image = self.rob.get_image_front()
        if isinstance(self.rob, SimulationRobobo):
            image = cv2.flip(image, 0)
        cv2.imwrite(str(FIGRURES_DIR / "pic.png"), image)
        image_values = pic_calcs(str(FIGRURES_DIR / "pic.png"))

        state_values = np.concatenate((ir_values[[7,4,5,6]], image_values))
        return np.array(state_values, dtype=np.float32)

    def get_reward(self, action, old_state, new_state):
        reward = 0
        # modify as required for openCV values in state
        IR = new_state[:4]
        old_CV = old_state[4:]
        new_CV = new_state[4:]

        '''
        If rob moves closer to block, reward
        If block reaches bottom of cam and stays there, fine.
        If block moves away, punish
        '''
        # calc for bottom dist
        if old_CV[0] < new_CV[0]:
            reward += new_CV[0] * 10
        elif new_CV[0] == 1 and old_CV[0] == 1:
            reward += 10
        elif old_CV[0] > new_CV[0]:
            reward += (new_CV[0] - old_CV[0]) * 50

        # calc for side to side dist
        old_distance = abs(old_CV[1] - 0.5)
        new_distance = abs(new_CV[1] - 0.5)

        '''
        Reward if block gets closer to middle
        Punish if it gets further from middle
        '''
        reward += (old_distance - new_distance) * 10  
    
        # Checks if x values are BIG. goal is to not punish for touching food but still account for hitting walls
        if np.sum(IR > 0.8) >= 1:
            reward -= 50

        # reward += (action[0] + action[1]) * 5

        # if action[0] < 0 and action[1] < 0:
        #     reward -= 10

        return reward

    def is_done(self, state, reward):
        if np.any(state >= 1.0) or np.isnan(reward):
            return True
        return False

    def render(self, mode='human'):
        pass

    def close(self):
        self.rob.stop_simulation()

robobo_instance = SimulationRobobo()
env = DummyVecEnv([lambda: RoboboEnv(robobo_instance)])
env = VecCheckNan(env, raise_exception=True)

policy_kwargs = dict(
    net_arch=dict(pi=[64, 64], vf=[128, 128]), 
    activation_fn=torch.nn.ReLU,
    normalize_images=True
)

model = PPO('MlpPolicy', env, verbose=1, learning_rate=1e-4, n_steps=2048, batch_size=64, n_epochs=10, clip_range=0.1, ent_coef=0.01, policy_kwargs=policy_kwargs)
# model = TD3('MlpPolicy', env, verbose=1, learning_rate=1e-4, batch_size=512)

max_episodes = 1000

print_episode_callback = PrintEpisodeCallback(verbose=1)
stop_training_callback = StopTrainingOnMaxEpisodes(max_episodes=max_episodes, verbose=1)
swa_callback = SWACallback(swa_start=5000, swa_freq=1000, verbose=1)
save_best_model_callback = SaveBestModelCallback(verbose=1)

callback = [print_episode_callback, stop_training_callback, swa_callback, save_best_model_callback]

model.learn(total_timesteps=10000, callback=callback)

save_location = FIGRURES_DIR / "ppo_robobo_100k.zip"
print(f'saving robo to: {save_location}')
model.save(save_location)

obs = env.reset()

def run_all_actions(rob):
    print("Finished!!")