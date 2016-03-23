import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style('darkgrid')


def show_log(log_path):
    logs = np.loadtxt(log_path, delimiter='\t', skiprows=3)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    l = -100
    # ax.plot(logs[l:, 0], logs[l:, 1], label='loss')
    ax.plot(logs[l:, 0], logs[l:, 2], label='batch-auc')
    ax.plot(logs[l:, 0], logs[l:, 3], label='eval-auc')
    ax.legend()
    ax.set_title(log_path)
    ax.set_xlabel('step')
    fig.canvas.draw()
    plt.show()


show_log('../log/Mon Mar 21 09:05:30 2016 LR')
