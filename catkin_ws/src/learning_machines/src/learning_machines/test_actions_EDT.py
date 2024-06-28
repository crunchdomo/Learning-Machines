#!/usr/bin/env python3

import random
from collections import deque # for generating trees
import copy                   # for deepcopying
from typing import List



from collections import deque, namedtuple
from itertools import count

import numpy as np
import os
import matplotlib.pyplot as plt

from robobo_interface import (
    IRobobo,
    SimulationRobobo,
    HardwareRobobo,
)

from tqdm import tqdm

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines3.common.callbacks import BaseCallback, StopTrainingOnMaxEpisodes
import pandas as pd
from .process_image import process_image as pic_calcs
import cv2
import json


from pathlib import Path
RESULTS_DIR = Path("/root/results")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)




def process_image(image_path, colour='green'):
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

# Define the Decision Tree model

num_variables = 3

class Node:
    def __init__(self, feature: int = None, split_val: float = None, action: list = None):
        # split node
        self.feature = feature
        self.split_val = split_val
        
        # leaf node
        self.action = action


class DecisionTree:
    def __init__(self, max_depth: int):
        self.max_depth = max_depth
        # store nodes in an array
        # left child = parent index * 2
        # right child = parent index * 2 + 1
        # root node = index 1
        l = 2 ** (max_depth + 1)
        self.nodes = [None for i in range(l)]
        
        # depths of each node
        self.depth_l = [0 for i in range(l)]
        # total depth of tree
        self.depth = 0
        
        # root node will be a split node
        root_feature = random.choice(range(num_variables)) # randomly choose a sensor
        self.nodes[1] = Node(
            feature=root_feature,
            split_val=random.uniform(0, 1)
        )
    
    def add_node(self, par_ind: int, node_type: str):
        if node_type == "leaf":
            new_node = Node(action=[random.uniform(-4000, 4000),random.uniform(-4000, 4000)])
        elif node_type == "split":
            feature = random.choice(range(num_variables)) # randomly choose a sensor
            new_node = Node(
                feature=feature,
                split_val=random.uniform(0, 1)
            )
        
        # check if left child exists
        if self.nodes[par_ind * 2] is None:
            next_pos = par_ind * 2
        else:
            next_pos = par_ind * 2 + 1
        
        self.nodes[next_pos] = new_node
        self.depth_l[next_pos] = self.depth_l[par_ind] + 1
        self.depth = max(self.depth, self.depth_l[next_pos])
        return next_pos
    
    def select_action(self, state: List):
        # classify one piece of data
        cur = 1
        while True:
            node = self.nodes[cur]
            
            # check if current node is a leaf node
            if node.action is not None:
                return node.action

            cur *= 2
            if state[node.feature] > node.split_val:
                cur += 1

    def to_dict(self):
        return {
            "max_depth": self.max_depth,
            "nodes": [
                {
                    "feature": node.feature,
                    "split_val": node.split_val,
                    "action": node.action
                } if node else None
                for node in self.nodes
            ],
            "depth_l": self.depth_l,
            "depth": self.depth
        }

    @staticmethod
    def from_dict(tree_dict):
        tree = DecisionTree(tree_dict["max_depth"])
        tree.nodes = [
            Node(
                feature=node["feature"],
                split_val=node["split_val"],
                action=node["action"]
            ) if node else None
            for node in tree_dict["nodes"]
        ]
        tree.depth_l = tree_dict["depth_l"]
        tree.depth = tree_dict["depth"]
        return tree

    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f)

    @staticmethod
    def load_from_file(filename):
        with open(filename, 'r') as f:
            tree_dict = json.load(f)
            return DecisionTree.from_dict(tree_dict)

# Randomly Generated Trees
# A breadth-first-search approach will be used to randomly generate the initial population. The split_p parameter is the probability that each randomly generated node will be a split node.

def generate_random(max_depth: int, split_p: float):
    ret = DecisionTree(max_depth)
    q = deque([1])
    while q:
        # index of the parent node
        cur = q.popleft()
        
        # loop twice for each child of the parent
        for _ in range(2):
            # make sure that we don't add a split node 
            # at the maximum depth
            if cur * 4 < len(ret.nodes) and split_p <= random.random():
                next_pos = ret.add_node(cur, "split")
                q.append(next_pos)
            else:
                ret.add_node(cur, "leaf")
                
    return ret

# Define helper functions


def get_state(rob, target_colour):
    #ir_values = rob.read_irs()
    #ir_values = np.clip(ir_values, 0, 10000)
    #ir_values = ir_values / 10000.0
    # CV values
    i = rob.get_image_front()
    image = cv2.flip(i, -1)
    # if isinstance(self.rob, SimulationRobobo):
    #     image = cv2.flip(image, 0)
    cv2.imwrite(str(FIGURES_DIR / "pic.png"), image)
    picture_data = process_image(str(FIGURES_DIR / "pic.png"), target_colour)
    #state_values = np.concatenate((picture_data, ir_values[[7,4,5,6]]))
    #return np.array(state_values, dtype=np.float32)
    return np.array(picture_data, dtype=np.float32)

    
def get_reward(state):
    weights = [0.5,1,0.5] 
    return sum(w * p for w, p in zip(weights, state[:3])) 


def fitness(individual: DecisionTree, rob: IRobobo ): # given a tree, make simulation
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
        rob.set_phone_tilt_blocking(100, 100)

    
    total_reward = 0
    done = False
    target_colour = 'red'
    state = get_state(rob,target_colour)

    for t in count():
        action = individual.select_action(state)  

        left_speed = action[0]
        right_speed = action[1]

        rob.move_blocking(left_speed, right_speed, 100)
        state = get_state(rob, target_colour)
        reward = get_reward(state)
        
        if reward >= 0.5 and target_colour == 'green':        
            reward = 1 + (1 - t*0.01)
            # print("DID IT!!!!")
            done = True

        # switch to find red
        if reward >= 0.2 and target_colour == 'red':
            reward = 1
            target_colour = 'green'
            #print("looking for Greeeeen!!!")

        total_reward += reward


        if t > 150 or done==True:
            if isinstance(rob, SimulationRobobo):
                rob.stop_simulation()
            break
    return total_reward

def selection(population: List[DecisionTree], fitness: List[float], k: int):
    inds = random.sample(range(len(population)), k)
    ind = max(inds, key=lambda i: fitness[i])
    p1 = population[ind]

    inds = random.sample(range(len(population)), k)
    ind = max(inds, key=lambda i: fitness[i])
    p2 = population[ind]
    
    return p1, p2

def crossover(p1: DecisionTree, p2: DecisionTree):
    def replace(source: DecisionTree, replace: DecisionTree, ind: int):
        # BFS to replace one node with another
        q = deque([ind])
        while q:
            cur = q.popleft()
            source.nodes[cur] = replace.nodes[cur]
            if source.nodes[cur].action is None:
                q.append(cur * 2)
                q.append(cur * 2 + 1)
    
        # clean unused nodes
        # let garbage collector do the heavy lifting
        for i in range(2, len(source.nodes)):
            if source.nodes[i // 2] is None:
                source.nodes[i] = None
    
    overlaps = [
        i
        for i in range(len(p1.nodes))
        if p1.nodes[i] is not None and p2.nodes[i] is not None
    ]
    
    c1 = copy.deepcopy(p1)
    ind = random.choice(overlaps)
    replace(c1, p2, ind)

    c2 = copy.deepcopy(p2)
    ind = random.choice(overlaps)
    replace(c2, p1, ind)
    
    return c1, c2

def mutate(tree: DecisionTree, scale=1):
    for _ in range(int(2*scale)):
        # select a random node
        valid = [i for i in range(len(tree.nodes)) if tree.nodes[i] is not None]
        ind = random.choice(valid)
        
        if tree.nodes[ind].action is None:
            # if selected node is a split node
            feature = random.choice(range(num_variables))
            tree.nodes[ind] = Node(
                feature = feature, 
                split_val = random.uniform(0, 1)
            )
        else:
            # if selected node is a leaf node
            old_action = tree.nodes[ind].action
            tree.nodes[ind] = Node(action = [value+random.uniform(-40*scale, 40*scale) for value in old_action])

def plot_rewards(average_fitnesses, max_fitnesses):
    plt.figure()
    plt.plot(average_fitnesses)
    plt.xlabel('Generation')
    plt.ylabel('Average fitness')
    plt.title('Average fitness per Generation')
    plt.savefig(FIGURES_DIR / f'average_fitnesses_EDT.png')
    plt.close()

    plt.figure()
    plt.plot(max_fitnesses)
    plt.xlabel('Generation')
    plt.ylabel('Max fitness')
    plt.title('Max fitness per Generation')
    plt.savefig(FIGURES_DIR / f'max_fitnesses_EDT.png')
    plt.close()

def run_training_simulation(max_depth: int, split_p: float, population_size: int, cross_p: float, mut_p: float, generation_cnt: int, rob, target_colour = 'red',continue_training=True):
    # initial population
    n = population_size
    n_previous = n
    if continue_training==True:
        population = []
        for i in range(n_previous):
            model_path = str(FIGURES_DIR)+'/best.model.EDT.top'+str(i+1)
            model = DecisionTree.load_from_file(model_path)
            population.append(model)
            print('model loaded'+str(i))
    else:
        population = [generate_random(max_depth, split_p) for _ in range(n)]

    

    average_fitnesses = []
    max_fitnesses = []

    # main loop
    for gen in range(generation_cnt):
        # select the best individuals from population
        fitnesses = [fitness(tree,rob) for tree in tqdm(population)]
        sorted_population = [tree for _, tree in sorted(zip(fitnesses, population), key=lambda x: x[0], reverse=True)]
        top_n_individuals = sorted_population[:n_previous]
        for i in range(n_previous):
            top_n_individuals[i].save_to_file(str(FIGURES_DIR)+'/best.model.EDT.top'+str(i+1))
        
        # time-dependent parameter control
        if gen <= 20:
            scale_cross = (20-gen)/100 # from +20% to +0%
            scale_mut = 10*(20-gen)/100 + 1 # from *300% to *100%
        else:
            scale_cross = 0
            scale_mut = 1


        # selection + crossover
        new_pop = []
        
        for _ in range(int(n * (cross_p + scale_cross) / 2)):
            p1, p2 = selection(population, fitnesses, 3) # third paramter can be changed
            c1, c2 = crossover(p1, p2)
            new_pop.extend((c1, c2))

        # elitism
        # fill new population with best individuals fom previous generation
        fp = sorted(
            zip(fitnesses, population), key=lambda x: x[0], reverse=True
        )
        new_pop.extend(fp[i][1] for i in range(n - len(new_pop)))
        
        # adaptive mutation
        for i in random.sample(range(n), int(n * mut_p * scale_mut)): # from 3*mut_p to 1*mut_p
            if i < n*cross_p/2:
                mutate(new_pop[i],scale_mut)
            else: # elite
                mutate(new_pop[i],1)
        
        population = new_pop
        
        # print stats
        average_fitness_value = sum(fitnesses) / n
        average_fitnesses.append(average_fitness_value)

        max_fitness_value = max(fitnesses)
        max_fitnesses.append(max_fitness_value)


        print(f"Generation:       {gen + 1}/{generation_cnt}")
        print(f"Average fitness: {average_fitness_value}")
        print(f"Max fitness: {max_fitness_value}")

        plot_rewards(average_fitnesses, max_fitnesses)

        if gen > 3 and (average_fitnesses[-1]-average_fitnesses[-2])/average_fitnesses[-2] <= 0.01: # when fitnesses converge
            if (max_fitnesses[-1]-max_fitnesses[-2])/max_fitnesses[-2] <= 0.01:
                break

    final_fitnesses = [fitness(tree,rob) for tree in population]
    sorted_population = [tree for _, tree in sorted(zip(fitnesses, population), key=lambda x: x[0], reverse=True)]

    top_n_individuals = sorted_population[:n_previous]
    for i in range(n_previous):
        top_n_individuals[i].save_to_file(str(FIGURES_DIR)+'/best.model.EDT.top'+str(i+1))

    return population, final_fitnesses


def run_trained_model(rob: IRobobo):
    n = 50
    #for i in range(n):
    for i in range(n):
        model_path = str(FIGURES_DIR)+'/best.model.EDT.top'+str(i+1)
        model = DecisionTree.load_from_file(model_path)
        fitness_ = fitness(model,rob)
        print('model tested: '+str(i+1))

        
    else:
        print("No saved model found. Please train the model first.")
        return



# Initialize the agent and run the simulation

rob = SimulationRobobo()


# Toggle between testing or running a model
train = False
def run_all_actions(rob: IRobobo):
    if train:
        population, final_fitnesses = run_training_simulation(6, 0.5, 50, 0.7, 0.2, 50, rob, continue_training=False)

        # max_depth: int, split_p: float, population_size: int, cross_p: float, mut_p: float, generation_cnt: int

    else:
        run_trained_model(rob)






# discarded


def get_state_task2(rob: IRobobo):
    # IR values
    ir_values = rob.read_irs()
    ir_values = np.clip(ir_values, 0, 200)

    # GCV values
    image = rob.get_image_front()
    if isinstance(rob, SimulationRobobo):
        image = cv2.flip(image, 0)
    cv2.imwrite(str(FIGURES_DIR / "pic.png"), image)
    image_values = [100*v for v in pic_calcs(str(FIGURES_DIR / "pic.png"))]

    state_values = np.concatenate((ir_values, image_values))
    return np.array(state_values, dtype=np.float32)


def get_reward_task2(action, old_state, new_state, left_speed, right_speed):
        reward = 0
        # modify as required for openCV values in state
        IR = new_state[:8]
        old_CV = old_state[8:]
        new_CV = new_state[8:]

        '''
        If rob moves closer to block, reward
        If block reaches bottom of cam and stays there, fine.
        If block moves away, punish
        '''
        # calc for bottom dist
        if old_CV[0] < new_CV[0]:
            reward += new_CV[0] * 0.1
        elif new_CV[0] == 100 and old_CV[0] == 100:
            reward += 10
        elif old_CV[0] > new_CV[0]:
            reward += (new_CV[0] - old_CV[0]) * 0.5

        # calc for side to side dist
        old_distance = abs(old_CV[1] - 50)
        new_distance = abs(new_CV[1] - 50)

        '''
        Reward if block gets closer to middle
        Punish if it gets further from middle
        '''
        reward += (old_distance - new_distance) * 0.1  

        # reward += (action[0] + action[1]) * 5
        # Checks if x values are BIG. 
        # goal is to not punish for touching food but still account for hitting walls
        if np.sum(IR[[7,4,5,6]] > 0.8*200) >= 1:
            reward -= 50

        if abs(left_speed)+abs(right_speed) < 50:
            reward -= 10

        return reward
