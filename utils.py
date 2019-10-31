"""
CMPUT 652, Fall 2019 - Assignment #2

__author__ = "Hager Radi"

utility functions for plotting
"""
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# plotting averaged runs with their standard deviations
def plot_means_1(returns_over_runs, runs, episodes):
    ep_returns_means = []
    ep_returns_stds = []

    run_3 = random.sample(range(0, runs), 3)
    run_10 = random.sample(range(0, runs), 10)

    fig, ax = plt.subplots(1)

    x = np.arange(1, episodes+1)

    # averaging 3 plots
    mean = np.mean(returns_over_runs[run_3], axis=0)
    std = np.std(returns_over_runs[run_3], axis=0)

    y =  sliding_window(mean, 100)

    ax.plot(x, y, lw=2, color='blue' , label='3 averaged runs')
    ax.fill_between(x, y-std , y+std, facecolor='blue', alpha=0.2)
    #
    # # averaging 10 plots
    mean = np.mean(returns_over_runs[run_10], axis=0)
    std = np.std(returns_over_runs[run_10], axis=0)

    y =  sliding_window(mean, 100)

    ax.plot(x, y, lw=2, color='green', label='10 averaged runs')
    ax.fill_between(x, y-std , y+std, facecolor='green', alpha=0.2)

    # averaging 30 plots
    mean = np.mean(returns_over_runs, axis=0)
    std = np.std(returns_over_runs, axis=0)

    y =  sliding_window(mean, 100)

    ax.plot(x, y, lw=2, color='red', label='30 averaged runs')
    ax.fill_between(x, y-std , y+std, facecolor='red', alpha=0.2)

    # drawing all in one figure
    ax.set_title("Episode Return")
    ax.set_ylabel("Average Return (Sliding Window 100)")
    ax.set_xlabel("Episode")
    ax.set_title('Average runs with their standard deviation')
    ax.legend(loc = 'best')

    # plt.show()
    fig.savefig('1.png')
    plt.close(fig)

# plotting averaged runs with their standard error
def plot_means_2(returns_over_runs, runs, episodes):
    ep_returns_means = []
    ep_returns_stds = []

    run_3 = random.sample(range(0, runs), 3)
    run_10 = random.sample(range(0, runs), 10)

    fig, ax = plt.subplots(1)

    x = np.arange(1, episodes+1)

    # averaging 3 plots
    mean = np.mean(returns_over_runs[run_3], axis=0)
    std = np.std(returns_over_runs[run_3], axis=0) / np.sqrt(3)

    y =  sliding_window(mean, 100)

    ax.plot(x, y, lw=2, color='blue' , label='3 averaged runs')
    ax.fill_between(x, y-std , y+std, facecolor='blue', alpha=0.2)

    # averaging 10 plots
    mean = np.mean(returns_over_runs[run_10], axis=0)
    std = np.std(returns_over_runs[run_10], axis=0) / np.sqrt(10)

    y =  sliding_window(mean, 100)

    ax.plot(x, y, lw=2, color='green', label='10 averaged runs')
    ax.fill_between(x, y-std , y+std, facecolor='green', alpha=0.2)

    # averaging 30 plots
    mean = np.mean(returns_over_runs, axis=0)
    std = np.std(returns_over_runs, axis=0)/ np.sqrt(30)

    y =  sliding_window(mean, 100)

    ax.plot(x, y, lw=2, color='red', label='30 averaged runs')
    ax.fill_between(x, y-std , y+std, facecolor='red', alpha=0.2)

    # drawing all in one figure
    ax.set_title("Episode Return")
    ax.set_ylabel("Average Return (Sliding Window 100)")
    ax.set_xlabel("Episode")
    ax.set_title('Average runs with their standard error')
    ax.legend(loc = 'lower right')

    # plt.show()
    fig.savefig('2.png')
    plt.close(fig)

# plotting averaged runs with their min-max value (for self-check)
def plot_means_3(returns_over_runs, runs, episodes):
    ep_returns_means = []
    ep_returns_stds = []

    run_3 = random.sample(range(0, runs), 3)
    run_10 = random.sample(range(0, runs), 10)

    fig, ax = plt.subplots(1)

    x = np.arange(1, episodes+1)

    # averaging 3 plots
    mean = np.mean(returns_over_runs[run_3], axis=0)
    min = np.min(returns_over_runs[run_3], axis=0)
    max = np.max(returns_over_runs[run_3], axis=0)

    y =  sliding_window(mean, 100)

    ax.plot(x, y, lw=2, color='blue' , label='3 averaged runs')
    ax.fill_between(x, min , max, facecolor='blue', alpha=0.2)

    # averaging 10 plots
    mean = np.mean(returns_over_runs[run_10], axis=0)
    min = np.min(returns_over_runs[run_10], axis=0)
    max = np.max(returns_over_runs[run_10], axis=0)

    y =  sliding_window(mean, 100)

    ax.plot(x, y, lw=2, color='green', label='10 averaged runs')
    ax.fill_between(x, min , max, facecolor='green', alpha=0.2)

    # averaging 30 plots
    mean = np.mean(returns_over_runs, axis=0)
    min = np.min(returns_over_runs, axis=0)
    max = np.max(returns_over_runs, axis=0)

    y =  sliding_window(mean, 100)

    ax.plot(x, y, lw=2, color='red', label='30 averaged runs')
    ax.fill_between(x, min , max, facecolor='red', alpha=0.2)

    # drawing all in one figure
    ax.set_title("Episode Return")
    ax.set_ylabel("Average Return (Sliding Window 100)")
    ax.set_xlabel("Episode")
    ax.set_title('Average runs with their min-max value')
    ax.legend(loc = 'best')

    # plt.show()
    fig.savefig('3.png')
    plt.close(fig)

# plotting 3 averaged runs, 10 times
def plot_means_4(returns_over_runs, runs, episodes):
    ep_returns = []

    indices = list(range(0,runs)) # list of integers from 0 to 29

    i = 0
    while i < 30:
        ep_returns.append(list(np.mean(returns_over_runs[i:i+2], axis=0)))
        i += 3

    fig, ax = plt.subplots(1)
    for means in ep_returns:
        ax.plot(np.arange(1, episodes+1) , means, lw=0.5)

    ax.set_title("Sampling and avergaing 3 runs (10 times)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Return (Sliding Window 100)")
    # plt.show()

    fig.savefig('4.png')
    plt.close(fig)

# plotting 30 averaged runs with their standard error
def plot_means_5(data1, data2, runs, episodes):
    ep_returns_means = []
    ep_returns_stds = []

    fig, ax = plt.subplots(1)

    x = np.arange(1, episodes+1)

    # averaging 30 plots
    mean = np.mean(data1, axis=0)
    std = np.std(data1, axis=0)/ np.sqrt(30)

    y =  sliding_window(mean, 100)

    ax.plot(x, y, lw=1, color='red',label='Variant')
    ax.fill_between(x, y-std , y+std, facecolor='red', alpha=0.2)

    mean = np.mean(data2, axis=0)
    std = np.std(data2, axis=0)/ np.sqrt(30)

    y =  sliding_window(mean, 100)

    ax.plot(x, y, lw=1, color='blue', label='Baseline')
    ax.fill_between(x, y-std , y+std, facecolor='blue', alpha=0.2)
    # drawing all in one figure
    ax.set_title("Episode Return")
    ax.set_ylabel("Average Return (Sliding Window 100)")
    ax.set_xlabel("Episode")
    ax.set_title('Average runs with their standard error')
    ax.legend(loc = 'lower right')

    # plt.show()
    fig.savefig('5.png')
    plt.close(fig)

if __name__ == '__main__':
    data1 = np.load('returns_30runs_variant.npy')
    data2 = np.load('returns_30runs_baseline.npy')
    # print(data.shape)

    runs = 30
    episodes = 2000

    plot_means_1(data2, runs, episodes)
    plot_means_2(data2, runs, episodes)
    plot_means_3(data2, runs, episodes)
    #
    # plot_means_4(data2, runs, episodes)
    plot_means_5(data1, data2, runs, episodes)
