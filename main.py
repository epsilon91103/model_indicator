import json

import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = (8, 15)


def get_mean(items, size):
    x_ = np.array(items[-size:])
    w = np.linspace(1, 0, num=size)
    return np.sum(x_*w) / np.sum(w)


def read_data(name_file='data.txt'):
    with open(name_file, 'r') as f:
        data_str = f.readline()
    data = json.loads(data_str)
    return data


def model_data(x, start_degradation, end_degradation):
    size_degradation = end_degradation - start_degradation

    data_before_deg = list(x[:start_degradation])
    data_after_deg = list(x[end_degradation:])
    data_after_deg_rev = data_after_deg[::-1]

    k = np.linspace(0, 1, num=size_degradation)
    k_r = 1 - k
    new_x = []
    for i in range(size_degradation):
        wma_a = get_mean(data_before_deg, size_degradation)
        wma_b = get_mean(data_after_deg_rev, size_degradation)
        new_x.append(wma_a * k_r[i] + wma_b * k[i])
        data_before_deg.append(wma_a)
        data_after_deg_rev.append(wma_b)

    new_x = list(x[:start_degradation]) + new_x + data_after_deg

    return new_x


def plot_and_save(lines, name_lines):
    for (x, y) in lines:
        plt.plot(x, y)

    plt.grid()
    plt.legend(name_lines)

    plt.xlabel('x')
    plt.ylabel('y')

    plt.savefig('plot')


def main():
    data = read_data()
    x, y = zip(*data)

    start_degradation = 828
    end_degradation = 890

    new_x = model_data(x, start_degradation, end_degradation)
    lines = [
        [x, y],
        [new_x, y]
    ]
    name_lines = [
        'default',
        'model'
    ]

    plot_and_save(lines, name_lines)


if __name__ == '__main__':
    main()
