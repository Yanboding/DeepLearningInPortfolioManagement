import matplotlib.pyplot as plt
import numpy as np


def set_fontsize(ax, size):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(size)


def plot_total_reward(total_rewards):
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.set_xlabel('Episode')
    ax.set_ylabel("Total Reward")
    ax.plot(np.arange(len(total_rewards)), total_rewards, '-')
    set_fontsize(ax, 20)
    plt.show()


def plot_confidence_intervals(confidence_intervals, includeText=False):
    '''
    confidence_intervals:
    {'x':{'xlabel': str, 'data': list(float)},
     'y': {'ylabel': str, 'data': list({'sample_mean': list(float),
                                        'half_window': list(float),
                                        'label': str})
        }
    }
    '''
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.set_xlabel(confidence_intervals['x']['xlabel'])
    ax.set_ylabel(confidence_intervals['y']['ylabel'])

    for d in confidence_intervals['y']['data']:
        d['sample_mean'] = np.array(d['sample_mean'])
        d['half_window'] = np.array(d['half_window'])
        ax.plot(confidence_intervals['x']['data'], d['sample_mean'], '--', label=d['label'])
        if includeText:
            for xdata, ydata, yhw in zip(confidence_intervals['x']['data'], d['sample_mean'], d['half_window']):
                ax.text(xdata, ydata, '{:g}'.format(float('{:.2g}'.format(ydata))) + '+-' + '{:g}'.format(
                    float('{:.1g}'.format(yhw))), color="red", fontsize=12)
        ax.fill_between(confidence_intervals['x']['data'], (d['sample_mean'] - d['half_window']),
                        (d['sample_mean'] + d['half_window']), alpha=.1)
    set_fontsize(ax, 20)
    plt.legend(fontsize=13)
    plt.show()

def plot_boxplot(data):
    '''
    {'x':{'xlabel':str, 'data':list(str) or None}, 'y':{'ylabel':str, 'data'[list(float)]}}
    '''
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.set_xlabel(data['x']['xlabel'])
    ax.set_ylabel(data['y']['ylabel'])
    plt.boxplot(data['y']['data'])
    if 'data' in data['x'] and data['x']['data']:
        ax.set_xticks([i for i in range(1, len(data['x']['data'])+1)], data['x']['data'])
    set_fontsize(ax, 20)
    plt.show()
