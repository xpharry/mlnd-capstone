# plot reward history

import csv
import matplotlib.pyplot as plt


def load_data(fname):
    x = []
    y = []
    with open(fname, 'r') as rfile:
        reward_reader = csv.reader(rfile, delimiter=' ')
        for row in reward_reader:
            # print(row)
            x.append(int(row[0]))
            y.append(float(row[1]))
    return x, y


def main():
    csv1 = '../result/dqn_reward_history.csv'
    csv2 = '../result/random_reward_history.csv'

    timesteps1, rewards1 = load_data(csv1)
    timesteps2, rewards2 = load_data(csv2)

    nsteps = min(len(timesteps1), len(timesteps2))
    print("timesteps length= %d" % nsteps)

    plt.figure(1)
    plt.plot(range(nsteps), rewards1[:nsteps], 'r*')
    plt.plot(range(nsteps), rewards2[:nsteps], 'g*')
    plt.xlabel('Time Steps')
    plt.ylabel('Rewards')
    plt.show()


if __name__ == "__main__":
    main()
