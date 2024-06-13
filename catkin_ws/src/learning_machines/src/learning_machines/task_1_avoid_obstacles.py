#!/usr/bin/env python3
import sys
from robobo_interface import SimulationRobobo, HardwareRobobo
from task_1_model import Model, ReplayMemory, optimize_model, tau, select_action
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from tqdm import trange
from itertools import count


action_space = ['move_forward', 'move_backward', 'turn_left', 'turn_right']
num_episodes = 1000 
collision_threshold = 150 # sensor reading to stop episode for collision
print_every = 50 # print results and save every X episodes
training = True
max_time = 200 # max number of actions per episode

np.random.seed(0xC0FFEE)
torch.manual_seed(0xC0FFEE)

def do_action(rob, action_idx, action_space):
    action = action_space[action_idx]
    if action == 'move_forward':
        rob.move_blocking(20, 20, 200)
    elif action == 'move_backward':
        rob.move_blocking(-20, -20, 200)
    elif action == 'turn_left':
        rob.move_blocking(-20, 20, 200)
    elif action == 'turn_right':
        rob.move_blocking(20, -20, 200)
    else:
        print('unknown action:', action_idx)


def collided(rob) -> bool:
    return any([r > collision_threshold for r in rob.read_irs()])


def target_function(rob, action):
    # we encourage robot to move forward, but also ok with other actions
    # a thing here is that he may go forward-backward all the time
    if collided(rob):
        return -100
    if action == 0:
        return 1
    if action == 1:
        return 0.3
    if action == 2 or action == 3:
        return 0.5 


if __name__ == "__main__":
    if sys.argv[1] == "--hardware":
        rob = HardwareRobobo(camera=True)
    elif sys.argv[1] == "--simulation":
        rob = SimulationRobobo()
        rob.play_simulation()
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument, --simulation or --hardware expected.")
    
    if sys.argv[2] == '--train':
        training = True
    elif sys.argv[2] == '--test':
        training = False
    else:
        raise ValueError(f"{sys.argv[2]} is not a valid argument, --train or --test expected.")

    print(f'running {"training" if training else "inference"} in {"real world" if isinstance(rob, HardwareRobobo)else "simulation"}')

    if training and isinstance(rob, HardwareRobobo):
        raise ValueError('Cannot train in real life!')

    init_pos = rob.get_position()
    init_rot = rob.read_orientation()
    if training:
        policy_net = Model(action_space)
        target_net = Model(action_space)
        
        policy_net.load_state_dict(torch.load('/root/results/task1_checkpoint_200.pth'))
        target_net.load_state_dict(policy_net.state_dict())

        optimizer = optim.AdamW(policy_net.parameters(), lr=1e-3, amsgrad=True)
        memory = ReplayMemory(10000)
        scores = []


        for i_episode in trange(200, num_episodes):
            # rob.set_position(init_pos, init_rot)
            rob.play_simulation()

            state = rob.read_irs()

            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            for t in count():
                action = select_action(policy_net, state, i_episode)
                do_action(rob, action, action_space)
                observation, reward, terminated = rob.read_irs(), target_function(rob, action), collided(rob)
                reward = torch.tensor([reward])
                done = terminated or t == max_time

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

                memory.push(state, action, next_state, reward)

                state = next_state

                optimize_model(memory, policy_net, target_net, optimizer)

                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
                target_net.load_state_dict(target_net_state_dict)

                if done:
                    scores.append(t + 1)
                    rob.stop_simulation()
                    break


            if i_episode % print_every == 0:
                print(f'Episode {i_episode}\tLast Score: {scores[-1]})')
                policy_net.save_checkpoint(f'/root/results/task1_checkpoint_{i_episode}.pth')
                
                plt.plot(range(len(scores)), scores)
                plt.title('Score per training episode')
                plt.savefig('/root/results/figures/task1.png')


    else:
        model = Model(action_space)
        # model.load_state_dict(torch.load('/root/results/task1_checkpoint_300.pth'))
        ep_len = 0
        while not collided(rob):
            observation = rob.read_irs()
            action = model.predict(observation)
            do_action(rob, action, action_space)
            ep_len += 1

        print(f'Episode length: {ep_len} actions')

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()