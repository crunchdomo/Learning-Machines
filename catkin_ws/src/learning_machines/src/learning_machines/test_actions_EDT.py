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
from data_files import FIGRURES_DIR
from robobo_interface import (
    IRobobo,
    SimulationRobobo,
    HardwareRobobo,
)



import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines3.common.callbacks import BaseCallback, StopTrainingOnMaxEpisodes
import pandas as pd
from .process_image import process_image as pic_calcs
import cv2
import json


# Define the Decision Tree model

num_variables = 6

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
        
        # store nodes in an array to avoid
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
            split_val=random.uniform(0, 75)
        )
    
    def add_node(self, par_ind: int, node_type: str):
        if node_type == "leaf":
            new_node = Node(action=[random.uniform(-100, 100),random.uniform(-100, 100)])
        elif node_type == "split":
            feature = random.choice(range(num_variables)) # randomly choose a sensor
            new_node = Node(
                feature=feature,
                split_val=random.uniform(0, 75)
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
def get_state(rob: IRobobo):
    # IR values
    ir_values = rob.read_irs()
    ir_values = np.clip(ir_values, 0, 10000)
    ir_values = ir_values / 10000.0

    # GCV values
    image = rob.get_image_front()
    if isinstance(rob, SimulationRobobo):
        image = cv2.flip(image, 0)
    cv2.imwrite(str(FIGRURES_DIR / "pic.png"), image)
    image_values = pic_calcs(str(FIGRURES_DIR / "pic.png"))*100

    state_values = np.concatenate((ir_values[[7,4,5,6]], image_values))
    return np.array(state_values, dtype=np.float32)

def get_reward(action, old_state, new_state):
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

        # reward += (action[0] + action[1]) * 5
    
        # Checks if x values are BIG. 
        # goal is to not punish for touching food but still account for hitting walls
        if np.sum(IR > 0.8) >= 1:
            reward -= 50

        return reward

def fitness(individual: DecisionTree, rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    state = get_state(rob)
    total_reward = 0

    for t in count():
        action = individual.select_action(state)  

        left_speed = action[0]
        right_speed = action[1]

        old_state = get_state(rob)
        rob.move_blocking(left_speed, right_speed, 100)
        next_state = get_state(rob)
            
        reward = get_reward(action, old_state, next_state)
        total_reward += reward
        state = next_state

        if t > 50:
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

def mutate(tree: DecisionTree):
    # select a random node
    valid = [i for i in range(len(tree.nodes)) if tree.nodes[i] is not None]
    ind = random.choice(valid)
    
    if tree.nodes[ind].action is None:
        # if selected node is a split node
        feature = random.choice(range(num_variables))
        tree.nodes[ind] = Node(
            feature = feature, 
            split_val = random.uniform(0, 75)
        )
    else:
        # if selected node is a leaf node
        old_action = tree.nodes[ind].action
        tree.nodes[ind] = Node(action = [value+random.uniform(-20, 20) for value in old_action])

def plot_rewards(average_fitnesses, max_fitnesses):
    plt.figure()
    plt.plot(average_fitnesses)
    plt.xlabel('Generation')
    plt.ylabel('Average fitness')
    plt.title('Average fitness per Generation')
    plt.savefig(FIGRURES_DIR / f'average_fitnesses_EDT.png')
    plt.close()

    plt.figure()
    plt.plot(max_fitnesses)
    plt.xlabel('Generation')
    plt.ylabel('Max fitness')
    plt.title('Max fitness per Generation')
    plt.savefig(FIGRURES_DIR / f'max_fitnesses_EDT.png')
    plt.close()

def run_training_simulation(max_depth: int, split_p: float, population_size: int, cross_p: float, mut_p: float, generation_cnt: int, rob):
    # initial population
    n = population_size
    population = [generate_random(max_depth, split_p) for _ in range(n)]
    average_fitnesses = []
    max_fitnesses = []

    # main loop
    for gen in range(generation_cnt):
        # select the best individuals from population
        fitnesses = [fitness(tree,rob) for tree in population]
        
        # selection + crossover
        new_pop = []
        for _ in range(int(n * cross_p / 2)):
            p1, p2 = selection(population, fitnesses, 3) # third paramter can be changed
            c1, c2 = crossover(p1, p2)
            new_pop.extend((c1, c2))

        # elitism
        # fill new population with best individuals fom previous generation
        fp = sorted(
            zip(fitnesses, population), key=lambda x: x[0], reverse=True
        )
        new_pop.extend(fp[i][1] for i in range(n - len(new_pop)))
        
        # mutation
        for i in random.sample(range(n), int(n * mut_p)):
            mutate(new_pop[i])
        
        population = new_pop
        
        # print stats
        average_fitness_value = sum(fitnesses) / n
        average_fitnesses.append(average_fitness_value)

        max_fitness_value = max(fitnesses)
        max_fitnesses.append(max_fitness_value)


        print(f"Generation:       {gen + 1}/{generation_cnt}")
        print(f"Average fitness: {average_fitness_value}")
        print(f"Max fitness: {max_fitness_value}")

        if gen > 3 and (average_fitnesses[-1]-average_fitnesses[-2])/average_fitnesses[-2] <= 0.01: # when fitnesses converge
            break

    plot_rewards(average_fitnesses, max_fitnesses)
    final_fitnesses = [fitness(tree,rob) for tree in population]

    return population, final_fitnesses



def run_trained_model(rob: IRobobo, model_path = 'best.model.EDT.top1'):
    tree = DecisionTree.load_from_file(model_path)
    if os.path.exists(model_path):
        fitness = fitness(tree,rob)
        
    else:
        print("No saved model found. Please train the model first.")
        return



# Initialize the agent and run the simulation

rob = SimulationRobobo()


# Toggle between testing or running a model
train = True
def run_all_actions(rob: IRobobo):
    if train:
        population, final_fitnesses = run_training_simulation(7, 0.5, 100, 0.7, 0.3, 50, rob)
        # max_depth: int, split_p: float, population_size: int, cross_p: float, mut_p: float, generation_cnt: int
        sorted_population = [tree for _, tree in sorted(zip(final_fitnesses, population), reverse=True)]
        print(final_fitnesses)
        top_5_individuals = sorted_population[:5]
        for i in range(5):
            top_5_individuals[i].save_to_file(str(FIGRURES_DIR)+'/best.model.EDT.top'+str(i+1))

    else:
        run_trained_model(rob, model_path = str(FIGRURES_DIR)+'best.model.EDT.top1')


