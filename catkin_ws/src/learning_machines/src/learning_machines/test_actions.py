#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import numpy as np
import os
from itertools import count
import math
import cv2
import datetime

from typing import Literal

from data_files import FIGRURES_DIR

from robobo_interface import (
        IRobobo,
        SimulationRobobo
)

# Helper fns
def get_state(rob: IRobobo, clamp = 250) -> torch.Tensor:
    # GCV values
    image = rob.get_image_front()
    if isinstance(rob, SimulationRobobo):
        image = cv2.flip(image, 0)
    cv2.imwrite(str(FIGRURES_DIR / "pic.png"), image)
    green_values = process_image(str(FIGRURES_DIR / "pic.png"), 'green', rob)
    red_values = process_image(str(FIGRURES_DIR / "pic.png"), 'red', rob)

    # IR values
    # ir_values = rob.read_irs()
    # ir_values = np.clip(ir_values, 0, 10000)
    # ir_values = ir_values / 10000.0
    ir_values = [list(map(lambda ir: ir if ir < clamp else clamp, rob.read_irs()))]

    # Select only IR values 7, 4, 5, 6
    selected_ir_values = [ir_values[0][i] for i in [7, 4, 5, 6]]

    # Combine all values
    combined_values = selected_ir_values + list(green_values) + list(red_values)

    # Convert to torch tensor
    return torch.tensor([combined_values], device=device, dtype=torch.float)
    # return torch.tensor([list(map(lambda ir: ir if ir < clamp else clamp, rob.read_irs()))], device=device, dtype=torch.float)

def get_device_type() -> Literal['cuda', 'mps', 'cpu']: 
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

device = torch.device(get_device_type())

Transition = namedtuple('Transition', ('state', 'speeds', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class WheelDQN(nn.Module):
    def __init__(self, n_observations):
        super(WheelDQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 2)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class RobotNNController:
    def __init__(self, n_observations, batch_size = 128, gamma = 0.99, lr=1e-4, memory_capacity=10000):
        self.steps_done = 0
        self.state_size = n_observations
        
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr

        self.policy_net = WheelDQN(n_observations).to(device)
        self.target_net = WheelDQN(n_observations).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayMemory(memory_capacity)

        self.target_net.eval()
        
        self.epsilon_start = 0.9
        self.epsilon_end = 0.05
        self.epsilon_decay = 1000 


    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        sample = random.random()
        self.epsilon_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if sample > self.epsilon_threshold:
            with torch.no_grad():
                return self.policy_net(state)
        else:
            return torch.tensor([[random.uniform(-50, 100), random.uniform(-50, 100)]], device=device, dtype=torch.float64)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def push(self, state, speeds, next_state, reward):
        self.memory.push(state, speeds, next_state, reward)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        
        state_speed_values = self.policy_net(state_batch)

        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_speed_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def process_image(image_path, colour='green', rob=SimulationRobobo()):
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

def get_reward(rob_after_movement, starting_pos, left_speed, right_speed, state, next_state, image_before_movement, image_after_movement, move_time):

    global food_consumed

    irs_after_movement = get_state(rob_after_movement)
    # print(irs_after_movement)
    current_pos = rob_after_movement.get_position()
    wheels = rob_after_movement.read_wheels()

    left = wheels.wheel_pos_l
    right = wheels.wheel_pos_r

    distance = np.linalg.norm(np.array([current_pos.x, current_pos.y]) - np.array([starting_pos.x, starting_pos.y]))

    reward = 0

    if food_consumed < rob_after_movement.nr_food_collected():
        reward += 0.2 * (rob_after_movement.nr_food_collected() - food_consumed) * (0 if left_speed < 0 and right_speed < 0 else 1)
        food_consumed = rob_after_movement.nr_food_collected()
            
    return torch.tensor([reward], device=device)

def run_training(rob: SimulationRobobo, controller: RobotNNController, num_episodes = 30, load_previous=False, moves=20):
    highest_reward = -float('inf')
    model_path = FIGRURES_DIR 

    total_left, total_right = 0.0, 0.0

    global rewards
    global sensor_readings
    sensor_readings = np.array([])
    rewards = np.array([])

    if load_previous and os.path.exists(model_path):
        controller.policy_net.load_state_dict(torch.load(model_path))
        controller.target_net.load_state_dict(controller.policy_net.state_dict())
        print("Loaded saved model.")

    for episode in range(num_episodes):
        # iterations_since_last_collision = 1

        print(f'Started Episode: {episode}')
        
        # Start the simulation
        rob.play_simulation()
        rob.set_phone_tilt_blocking(100, 100)
        rob.sleep(0.5)

        state = get_state(rob)
        camera_state = state[4:]
        starting_pos = rob.get_position()
        total_reward = 0

        global food_consumed
        food_consumed = 0

        for t in count():
            # state here is what we see before moving
            new_readings = rob.read_irs()
            sensor_readings = np.append(sensor_readings, new_readings)
            speeds = controller.select_action(state)
            left_speed, right_speed = speeds[0, 0].item(), speeds[0, 1].item() # choose a movement
            move_time = 100
            rob.reset_wheels()
            rob.move_blocking(int(left_speed), int(right_speed), move_time) # execute movement
            next_state = get_state(rob) # what we see after moving
            next_camera_state = next_state[4:]
            wheels = rob.read_wheels()

            total_left += wheels.wheel_pos_l
            total_right += wheels.wheel_pos_r
            
            # reward gets rob (after moving), left_speed and right_speed (of the last movement),
            reward = get_reward(rob, starting_pos, left_speed, right_speed, state, next_state, camera_state, next_camera_state, move_time)
            total_reward += reward.item()

            controller.push(state, speeds, next_state, reward)
            state = next_state
            camera_state = next_camera_state

            controller.optimize_model()

            if t > moves + (episode // 5) * 5 :
                rob.stop_simulation()
                break
        rewards = np.append(rewards, total_reward)
        controller.update_target()
        print(f"Episode: {episode}, Total Reward: {total_reward}")
        generate_plots()
        if total_reward > highest_reward:
            highest_reward = total_reward
            torch.save(controller.policy_net.state_dict(), model_path / f"goodest.test")
            print(f"Saved best model with highest reward: {highest_reward}")

def clamp(n, smallest, largest): 
    if n < 0:
        return max(n, smallest)
    return min(n, largest)

def run_model(rob: IRobobo, controller: RobotNNController):
    # load the model
    model_path = FIGRURES_DIR  / 'top_hardware.model'
    controller.policy_net.load_state_dict(torch.load(model_path))
    controller.target_net.load_state_dict(controller.policy_net.state_dict())
    controller.policy_net.eval()

    # Start the simulation
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    state = get_state(rob)
    
    collisions = 0
    still_colliding = False

    while True:
        speeds = controller.select_action(state)
        left_speed, right_speed = speeds[0, 0].item(), speeds[0, 1].item()
        print(f"Speeds: {left_speed}, {right_speed}")
        if isinstance(rob, SimulationRobobo):
            move_time = 100
        else:
            move_time = 500
        rob.reset_wheels()
        
        rob.move_blocking(clamp(int(left_speed), -100, 100), clamp(int(right_speed), -100, 100), move_time)
        next_state = get_state(rob)
        state = next_state

        if rob.read_irs()[0] > 250 and not still_colliding:
            collisions += 1
            still_colliding = True
            print(f"Collisions: {collisions}")
        elif rob.read_irs()[0] < 250:
            still_colliding = False

        # Exit on collision
        # if collisions > 9:
        #     break

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

controller = RobotNNController(n_observations=8, memory_capacity=10000, batch_size=64, gamma=0.99, lr=1e-3)

def generate_plots():
    global sensor_readings
    global rewards
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    sns.set_theme()
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(8):
        ax.plot(sensor_readings[i], label=f"IR {i+1}")

    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("IR Sensor Value")
    ax.set_title("IR Sensor Readings Over Time")
    
    # save the figure to the figures directory
    plt.savefig(FIGRURES_DIR / 'sensor_readings_training.png')

    # Plot the rewards
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(rewards)
    ax.set_xlabel("Time")
    ax.set_ylabel("Reward")
    ax.set_title("Rewards Over Time")
    plt.savefig(FIGRURES_DIR / 'rewards_training.png')

    plt.close()

def run_all_actions(rob):
    run_training(rob, controller, num_episodes=30, load_previous=False, moves=40)
    generate_plots()

def run_task1_actions(rob):
    run_model(rob, controller)

def run_task0_actions(rob):
    print('Task 0 actions')


# For Adam
rob = SimulationRobobo()
run_all_actions(rob)