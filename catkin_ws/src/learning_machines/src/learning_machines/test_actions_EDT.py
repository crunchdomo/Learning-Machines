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

# Define the Decision Tree model

actions = ['forward', 'turn_right', 'turn_left', 'back']

class Node:
    def __init__(self, feature: int = None, split_val: float = None, action: int = None):
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
        root_feature = random.choice(range(8)) # randomly choose a sensor
        self.nodes[1] = Node(
            feature=root_feature,
            split_val=random.uniform(0, 75)
        )
    
    def add_node(self, par_ind: int, node_type: str):
        if node_type == "leaf":
            new_node = Node(action=random.choice(actions))
        elif node_type == "split":
            feature = random.choice(range(8)) # randomly choose a sensor
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
    ir_values = rob.read_irs()
    return ir_values

def get_reward(start, p1, p2, movement, rob):
    reward = 0

    IRs = rob.read_irs()
    for ir in IRs:
        if ir > 1000:
            reward += 100
    if movement == 'forward' or movement == 'back':
        reward += 2

    return reward




def fitness(individual: DecisionTree, rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    state = get_state(rob)
    start_position = rob.get_position()
    total_reward = 0

    for t in count():
        p1 = rob.get_position()
        action = individual.select_action(state)  ##
        fastness = 50

        # Define movement based on single action value
        if action == 'forward':
            left_speed = fastness
            right_speed = fastness
        elif action == 'turn_right':
            left_speed = fastness
            right_speed = -fastness
        elif action == 'turn_left':
            left_speed = -fastness
            right_speed = fastness
        elif action == 'back':
            left_speed = -fastness
            right_speed = -fastness
        rob.move_blocking(left_speed, right_speed, 100)
        next_state = get_state(rob)
        p2 = rob.get_position()

            
        reward = get_reward(start_position, p1, p2, action, rob)
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
        feature = random.choice(range(8))
        tree.nodes[ind] = Node(
            feature = feature, 
            split_val = random.uniform(0, 75)
        )
    else:
        # if selected node is a leaf node
        tree.nodes[ind] = Node(action=random.choice(actions))


def run_training_simulation(max_depth: int, split_p: float, population_size: int, cross_p: float, mut_p: float, generation_cnt: int, rob):
    # initial population
    n = population_size
    population = [generate_random(max_depth, split_p) for _ in range(n)]
    
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
        print(f"Generation:       {gen + 1}/{generation_cnt}")
        print(f"Average fitness: {sum(fitnesses) / n}")
    
    return population






def plot_rewards(rewards, episode):
    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.savefig(FIGRURES_DIR / f'training_rewards_EDT.png')
    plt.close()




def run_trained_model(rob: IRobobo, agent):
    model_path = FIGRURES_DIR / 'best.model.EDT'
    
    if os.path.exists(model_path):
        pass
    else:
        print("No saved model found. Please train the model first.")
        return



# Initialize the agent and run the simulation

rob = SimulationRobobo()



# Toggle between testing or running a model
train = True
def run_all_actions(rob: IRobobo):
    if train:
        population = run_training_simulation(7, 0.5, 100, 0.7, 0.3, 50, rob)
    else:
        pass


