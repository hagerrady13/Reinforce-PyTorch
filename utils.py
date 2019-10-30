import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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

def plot_means_1(returns_over_runs, runs, episodes):
    ep_returns_means = []
    ep_returns_stds = []

    run_3 = random.sample(range(0, runs), 3)
    run_10 = random.sample(range(0, runs), 10)

    fig, ax = plt.subplots(1)

    x = np.arange(1, episodes+1)

    # averaging 3 plots
    # mean = np.mean(returns_over_runs[run_3], axis=0)
    # std = np.std(returns_over_runs[run_3], axis=0)
    #
    # y =  sliding_window(mean, 100)
    #
    # ax.plot(x, y, lw=2, color='blue' , label='3 averaged runs')
    # ax.fill_between(x, y-std , y+std, facecolor='blue', alpha=0.1)
    #
    # # averaging 10 plots
    # mean = np.mean(returns_over_runs[run_10], axis=0)
    # std = np.std(returns_over_runs[run_10], axis=0)
    #
    # y =  sliding_window(mean, 100)
    #
    # ax.plot(x, y, lw=2, color='green', label='10 averaged runs')
    # ax.fill_between(x, y-std , y+std, facecolor='green', alpha=0.1)

    # averaging 30 plots
    mean = np.mean(returns_over_runs, axis=0)
    std = np.std(returns_over_runs, axis=0)

    y =  sliding_window(mean, 100)

    ax.plot(x, y, lw=2, color='red', label='30 averaged runs')
    ax.fill_between(x, y-std , y+std, facecolor='red', alpha=0.1)

    # drawing all in one figure
    ax.set_title("Episode Return")
    ax.set_ylabel("Average Return (Sliding Window 100)")
    ax.set_xlabel("Episode")
    ax.set_title('Average runs with their standard deviation')
    ax.legend(loc = 'best')

    # plt.show()
    fig.savefig('1.png')
    plt.close(fig)

def plot_means_2(returns_over_runs, runs, episodes):
    ep_returns_means = []
    ep_returns_stds = []

    run_3 = random.sample(range(0, runs), 3)
    run_10 = random.sample(range(0, runs), 10)

    fig, ax = plt.subplots(1)

    x = np.arange(1, episodes+1)

    # averaging 3 plots
    # mean = np.mean(returns_over_runs[run_3], axis=0)
    # std = np.std(returns_over_runs[run_3], axis=0) / np.sqrt(3)
    #
    # y =  sliding_window(mean, 100)
    #
    # ax.plot(x, y, lw=2, color='blue' , label='3 averaged runs')
    # ax.fill_between(x, y-std , y+std, facecolor='blue', alpha=0.1)
    #
    # # averaging 10 plots
    # mean = np.mean(returns_over_runs[run_10], axis=0)
    # std = np.std(returns_over_runs[run_10], axis=0) / np.sqrt(10)
    #
    # y =  sliding_window(mean, 100)
    #
    # ax.plot(x, y, lw=2, color='green', label='10 averaged runs')
    # ax.fill_between(x, y-std , y+std, facecolor='green', alpha=0.1)

    # averaging 30 plots
    mean = np.mean(returns_over_runs, axis=0)
    std = np.std(returns_over_runs, axis=0)/ np.sqrt(30)

    y =  sliding_window(mean, 100)

    ax.plot(x, y, lw=2, color='red', label='30 averaged runs')
    ax.fill_between(x, y-std , y+std, facecolor='red', alpha=0.1)

    # drawing all in one figure
    ax.set_title("Episode Return")
    ax.set_ylabel("Average Return (Sliding Window 100)")
    ax.set_xlabel("Episode")
    ax.set_title('Average runs with their standard error')
    ax.legend(loc = 'best')

    # plt.show()
    fig.savefig('2.png')
    plt.close(fig)

def plot_means_3(returns_over_runs, runs, episodes):
    ep_returns_means = []
    ep_returns_stds = []

    run_3 = random.sample(range(0, runs), 3)
    run_10 = random.sample(range(0, runs), 10)

    fig, ax = plt.subplots(1)

    x = np.arange(1, episodes+1)

    # averaging 3 plots
    # mean = np.mean(returns_over_runs[run_3], axis=0)
    # min = np.min(returns_over_runs[run_3], axis=0)
    # max = np.max(returns_over_runs[run_3], axis=0)
    #
    # y =  sliding_window(mean, 100)
    #
    # ax.plot(x, y, lw=2, color='blue' , label='3 averaged runs')
    # ax.fill_between(x, min , max, facecolor='blue', alpha=0.1)
    #
    # # averaging 10 plots
    # mean = np.mean(returns_over_runs[run_10], axis=0)
    # min = np.min(returns_over_runs[run_10], axis=0)
    # max = np.max(returns_over_runs[run_10], axis=0)
    #
    # y =  sliding_window(mean, 100)
    #
    # ax.plot(x, y, lw=2, color='green', label='10 averaged runs')
    # ax.fill_between(x, min , max, facecolor='green', alpha=0.1)

    # averaging 30 plots
    mean = np.mean(returns_over_runs, axis=0)
    min = np.min(returns_over_runs, axis=0)
    max = np.max(returns_over_runs, axis=0)

    y =  sliding_window(mean, 100)

    ax.plot(x, y, lw=2, color='red', label='30 averaged runs')
    ax.fill_between(x, min , max, facecolor='red', alpha=0.1)

    # drawing all in one figure
    ax.set_title("Episode Return")
    ax.set_ylabel("Average Return (Sliding Window 100)")
    ax.set_xlabel("Episode")
    ax.set_title('Average runs with their min-max value')
    ax.legend(loc = 'best')

    # plt.show()
    fig.savefig('3.png')
    plt.close(fig)

def plot_means_4(returns_over_runs, runs, episodes):
    ep_returns = []

    indices = list(range(0,runs)) # list of integers from 0 to 29

    i = 0
    while i < 30:
        ep_returns.append(list(np.mean(returns_over_runs[i:i+2], axis=0)))
        i += 3

    fig, ax = plt.subplots(1)
    for means in ep_returns:
        ax.plot(np.arange(1, episodes+1) , means)

    ax.set_title("Sampling and avergaing 3 runs (10 times)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Return (Sliding Window 100)")
    # plt.show()

    fig.savefig('4.png')
    plt.close(fig)
