#!/usr/bin/env python3

"""
CMPUT 652, Fall 2019 - Assignment #2 solution - Hager Radi

__author__ = "Craig Sherstan"
__copyright__ = "Copyright 2019"
__credits__ = ["Craig Sherstan"]
__email__ = "sherstan@ualberta.ca"
"""
import torch
import matplotlib
import random
import matplotlib.pyplot as plt
import gym
from network import network_factory
import argparse
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import os
from utils import *

# prevents type-3 fonts, which some conferences disallow.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('TkAgg')

seed = 999

def make_env():
    env = gym.make('CartPole-v0')
    env.seed(seed)
    return env

if __name__ == '__main__':

    """
    python main.py --episodes 10000
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", "-e", default=5000, type=int, help="Number of episodes to train for")
    parser.add_argument("--gamma", "-g", default=1, type=int, help="Gamma")
    parser.add_argument("--timesteps", "-T", default=1000, type=int, help="Number of steps per episode")

    args = parser.parse_args()

    episodes = args.episodes
    gamma = args.gamma
    T = args.timesteps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env()
    # torch.manual_seed(seed)

    in_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    eps = np.finfo(np.float32).eps.item()

    # updating per each time step or per episode
    batch_update = True

    writer = SummaryWriter()
    runs = 1

    returns_over_runs = np.zeros(shape=(runs, episodes))
    # looping over different runs
    for run in range(runs):

        network = network_factory(in_size=in_size, num_actions=num_actions, env=env)
        network.to(device)

        print(network)
        # alpha_w, alpha_theta are the same
        optimizer = torch.optim.Adam(network.parameters(), lr=0.01)

        ep_returns = []

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

                state, reward, done, _  = env.step(action)
                total_reward += reward

                episode_reward.append(reward)
                log_probs.append(a_log_prob)
                state_values.append(state_value)

                if done:
                    break

                # env.render()

            if batch_update:
                G = []
                for r in episode_reward:
                    R = r + gamma * R
                    G.insert(0, R)

                G = torch.tensor(G)

                # apply whitening
                # G = (G - G.mean()) / (G.std() + eps) # To have small values of Loss

                p_losses  = []
                v_losses = []

                for a_log_prob, state_value, R in zip(log_probs, state_values, G):
                    p_losses.append(-1 * a_log_prob * (R - state_value.item()))
                    v_losses.append(torch.nn.functional.mse_loss(state_value, torch.tensor([R]), reduction='mean'))

                optimizer.zero_grad()

                policy_loss = torch.stack(p_losses).sum()
                value_loss =  torch.stack(v_losses).sum()
                loss = policy_loss + value_loss
                # print(loss.item())
                loss.backward()
                optimizer.step()

                # tensorboard Plotting
                if runs == 1:
                    writer.add_scalar('Loss/total', loss, ep)
                    writer.add_scalar('Loss/Policy', policy_loss, ep)
                    writer.add_scalar('Loss/StateValue', value_loss, ep)

                    writer.add_scalar('ValueModel/Linear1.weight', torch.mean(network.v1.weight.grad)**2, ep)
                    writer.add_scalar('ValueModel/Linear2.weight', torch.mean(network.v2.weight.grad)**2, ep)

                    writer.add_scalar('ValueModel/Linear1.bias', torch.mean(network.v1.bias.grad)**2, ep)
                    writer.add_scalar('ValueModel/Linear2.bias', torch.mean(network.v2.bias.grad)**2, ep)

                    writer.add_scalar('policy/Linear1.weight', torch.mean(network.p1.weight.grad)**2, ep)
                    writer.add_scalar('policy/Linear2.weight', torch.mean(network.p2.weight.grad)**2, ep)

                    writer.add_scalar('policy/Linear1.bias', torch.mean(network.p1.bias.grad)**2, ep)
                    writer.add_scalar('policy/Linear2.bias', torch.mean(network.p2.bias.grad)**2, ep)

            ep_returns.append(total_reward)

            returns_over_runs[run][ep] = total_reward

    if runs == 1:
        means = sliding_window(ep_returns, 100)
        plt.plot(means)
        plt.title("Episode Return")
        plt.xlabel("Episode")
        plt.ylabel("Average Return (Sliding Window 100)")
        plt.show()

        for ep in range(1, episodes+1):
            writer.add_scalar('Average Returns per episode', means[ep-1], ep)

    else:
        plot_means_1(returns_over_runs, runs, episodes)
        plot_means_2(returns_over_runs, runs, episodes)
        plot_means_3(returns_over_runs, runs, episodes)

        plot_means_4(returns_over_runs, runs, episodes)

    # save the trained network
    torch.save(network, 'model.pt')
    torch.save(network.state_dict(), 'checkpoint.pkl')
