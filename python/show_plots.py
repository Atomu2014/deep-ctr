import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

sns.set_style('darkgrid')
colors = sns.color_palette()


def show_log(log_path, mode, p1, p2):
    logs = np.loadtxt(log_path, delimiter='\t', skiprows=3)
    fig = plt.figure()
    ax0 = fig.add_subplot(211)
    ax1 = fig.add_subplot(212)

    if mode == 'all':
        ax0.plot(logs[:, 0], logs[:, 1], label='loss', color=colors[0])
        ax1.plot(logs[:, 0], logs[:, 2], label='batch-auc', color=colors[1])
        ax1.plot(logs[:, 0], logs[:, 3], label='eval-auc', color=colors[2])
    elif mode == 'head':
        ax0.plot(logs[:p1, 0], logs[:p1, 1], label='loss', color=colors[0])
        ax1.plot(logs[:p1, 0], logs[:p1, 2], label='batch-auc', color=colors[1])
        ax1.plot(logs[:p1, 0], logs[:p1, 3], label='eval-auc', color=colors[2])
    elif mode == 'tail':
        ax0.plot(logs[p2:, 0], logs[p2:, 1], label='loss', color=colors[0])
        ax1.plot(logs[p2:, 0], logs[p2:, 2], label='batch-auc', color=colors[1])
        ax1.plot(logs[p2:, 0], logs[p2:, 3], label='eval-auc', color=colors[2])
    else:
        ax0.plot(logs[p1:p2, 0], logs[p1:p2, 1], label='loss', color=colors[0])
        ax1.plot(logs[p1:p2, 0], logs[p1:p2, 2], label='batch-auc', color=colors[1])
        ax1.plot(logs[p1:p2, 0], logs[p1:p2, 3], label='eval-auc', color=colors[2])

    ax0.legend()
    ax0.set_title(log_path)
    ax0.set_xlabel('step')
    ax1.legend()
    ax1.set_title(log_path)
    ax1.set_xlabel('step')
    fig.canvas.draw()
    plt.show()


# show_log('../log/Wed Mar 23 14:43:38 2016 FM2')

if __name__ == '__main__':
    assert len(sys.argv) > 1, 'must input log path'
    log_path = ' '.join(sys.argv[1:-3])
    if sys.argv[-3] == 'all':
        show_log('../log/' + log_path, sys.argv[-3], None, None)
    elif sys.argv[-3] == 'head':
        show_log('../log/' + log_path, sys.argv[-3], int(sys.argv[-2]), None)
    elif sys.argv[-3] == 'tail':
        show_log('../log/' + log_path, sys.argv[-3], None, int(sys.argv[-1]))
    else:
        show_log('../log/' + log_path, sys.argv[-3], int(sys.argv[-2]), int(sys.argv[-1]))
