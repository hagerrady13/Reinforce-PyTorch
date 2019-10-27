#!/usr/bin/env python3

"""
CMPUT 652, Fall 2019 - Assignment #2

__author__ = "Craig Sherstan"
__copyright__ = "Copyright 2019"
__credits__ = ["Craig Sherstan"]
__email__ = "sherstan@ualberta.ca"
"""

"""
You are free to additional imports as needed... except please do not add any additional packages or dependencies to
your virtualenv other than those specified in requirements.txt. If I can't run it using the virtualenv I specified,
without any additional installs, there will be a penalty.

I've included a number of imports that I think you'll need.
"""
import torch
import matplotlib
import matplotlib.pyplot as plt
import gym
from network import network_factory
import argparse
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import os

# prevents type-3 fonts, which some conferences disallow.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('TkAgg')


def make_env():
    env = gym.make('CartPole-v0')
    return env


def sliding_window(data, N):
    """
    For each index, k, in data we average over the window from k-N-1 to k. The beginning handles incomplete buffers,
    that is it only takes the average over what has actually been seen.
    :param data: A numpy array, length M
    :param N: The length of the sliding window.
    :return: A numpy array, length M, containing smoothed averaging.
    """

    idx = 0
    window = np.zeros(N)
    smoothed = np.zeros(len(data))

    for i in range(len(data)):
        window[idx] = data[i]
        idx += 1

        smoothed[i] = window[0:idx].mean()

        if idx == N:
            window[0:-1] = window[1:]
            idx = N - 1

    return smoothed



if __name__ == '__main__':

    """
    You are free to add additional command line arguments, but please ensure that the script will still run with:
    python main.py --episodes 10000
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", "-e", default=1000, type=int, help="Number of episodes to train for")
    parser.add_argument("--gamma", "-g", default=1, type=int, help="Gamma")
    parser.add_argument("--timesteps", "-T", default=1000, type=int, help="Number of steps per episode")

    args = parser.parse_args()

    episodes = args.episodes
    gamma = args.gamma
    T = args.timesteps

    """
    It is unlikely that the GPU will help in this instance (since the size of individual operations is small) - in fact
    there's a good chance it could slow things down because we have to move data back and forth between CPU and GPU.
    Regardless I'm leaving this in here. For those of you with GPUs this will mean that you will need to move your
    tensors to GPU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env()

    in_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    network = network_factory(in_size=in_size, num_actions=num_actions, env=env)
    network.to(device)

    print(network)
    # alpha_w, alpha_theta are the same
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
    # TODO: after each episode add the episode return (the total sum of the undiscounted rewards received in the
    #  episode) to this list.
    ep_returns = []

    eps = np.finfo(np.float32).eps.item()

    # updating per each time step or per episode
    batch_update = False

    writer = SummaryWriter()

    for ep in range(episodes):
        episode_reward = []
        log_probs = []
        state_values = []

        R = 0
        total_reward = 0

        print("############### Starting Episode: " , ep)

        state = env.reset()
        for t in range(1, T):
            action, a_log_prob, state_value = network.get_action(torch.from_numpy(state).float().unsqueeze(0))
            # print(action)
            state, reward, done, _  = env.step(action)
            total_reward += reward

            episode_reward.append(reward)
            log_probs.append(a_log_prob)
            state_values.append(state_value)

            if done:
                break

            # env.render()

            if not batch_update:
                R = reward + (gamma * R)
                policy_loss = -1* a_log_prob * (R - state_value.item())
                value_loss = torch.nn.functional.mse_loss(state_value, torch.tensor([R]), reduction='mean')

                loss = policy_loss + value_loss

                print(loss.item())

                loss.backward()
                optimizer.step()

        if batch_update:
            G = []
            for r in episode_reward:
                R = r + gamma * R
                G.insert(0, R)

            # print("before", G[0])
            G = torch.tensor(G)

            # G = (G - G.mean()) / (G.std() + eps) # To have small values of Loss
            # print("after", G[0])

            p_losses  = []
            v_losses = []

            for a_log_prob, state_value, R in zip(log_probs, state_values, G):
                p_losses.append(-1 * a_log_prob * (R - state_value.item()))
                v_losses.append(torch.nn.functional.mse_loss(state_value, torch.tensor([R]), reduction='mean'))

            optimizer.zero_grad()

            loss = torch.stack(p_losses).sum() + torch.stack(v_losses).sum()
            # print(loss.item())
            loss.backward()
            optimizer.step()

        # print("Final Episode Return: ", episode_reward/t)
        # running_reward = 0.05 * total_reward + (1 - 0.05) * total_reward

        ep_returns.append(total_reward)

        # log results
        # print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(ep, total_reward, running_reward))
        #
        # # check if we have "solved" the cart pole problem
        # if running_reward > env.spec.reward_threshold:
        #     print("Solved! Running reward is now {} and "
        #           "the last episode runs to {} time steps!".format(running_reward, t))
        #     break

    # writer.add_scalar('Average Returns per episode', sliding_window(ep_returns, 100), episodes)

    plt.plot(sliding_window(ep_returns, 100))
    plt.title("Episode Return")
    plt.xlabel("Episode")
    plt.ylabel("Average Return (Sliding Window 100)")

    plt.show()

    # TODO: save your network
    torch.save(network, 'model.pt')
