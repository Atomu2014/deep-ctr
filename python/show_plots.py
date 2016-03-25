import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

sns.set_style('darkgrid')
colors = sns.color_palette()


def show_log(log_path, mode, p1, p2, p3, p4):
    logs = np.loadtxt(log_path, delimiter='\t', skiprows=3)
    fig = plt.figure()
    ax0 = fig.add_subplot(211)
    ax1 = fig.add_subplot(212)

    if mode == 'all':
        step = logs[::p3, 0]
        loss = logs[::p3, 1]
        batch_auc = logs[::p3, 2]
        eval_auc = logs[::p3, 3]
    elif mode == 'head':
        step = logs[:p1:p3, 0]
        loss = logs[:p1:p3, 1]
        batch_auc = logs[:p1:p3, 2]
        eval_auc = logs[:p1:p3, 3]
    elif mode == 'tail':
        step = logs[p2::p3, 0]
        loss = logs[p2::p3, 1]
        batch_auc = logs[p2::p3, 2]
        eval_auc = logs[p2::p3, 3]
    else:
        step = logs[p1:p2:p3, 0]
        loss = logs[p1:p2:p3, 1]
        batch_auc = logs[p1:p2:p3, 2]
        eval_auc = logs[p1:p2:p3, 3]

    smooth_auc = np.array(eval_auc[p4 - 1:])
    for i in range(p4 - 1):
        smooth_auc += eval_auc[i:(i - p4 + 1)]
    smooth_auc /= p4

    ax0.plot(step, loss, label='loss', color=colors[0])
    ax1.plot(step, batch_auc, label='batch-auc', color=colors[1])
    ax1.plot(step, eval_auc, label='eval-auc', color=colors[2])
    ax1.plot(step[p4 - 1:], smooth_auc, label='smoothed eval-auc', color=colors[3])

    ax0.legend()
    ax0.set_title(log_path)
    ax0.set_xlabel('step')
    ax1.legend()
    ax1.set_title(log_path)
    ax1.set_xlabel('step')
    fig.canvas.draw()
    plt.show()


if __name__ == '__main__':
    assert len(sys.argv) > 1, 'must input log path'
    log_path = ' '.join(sys.argv[1:-5])
    mode = sys.argv[-5]
    if mode == 'all':
        show_log('../log/' + log_path, mode, None, None, int(sys.argv[-2]), int(sys.argv[-1]))
    elif mode == 'head':
        show_log('../log/' + log_path, mode, int(sys.argv[-4]), None, int(sys.argv[-2]), int(sys.argv[-1]))
    elif mode == 'tail':
        show_log('../log/' + log_path, mode, None, int(sys.argv[-3]), int(sys.argv[-2]), int(sys.argv[-1]))
    else:
        show_log('../log/' + log_path, mode, int(sys.argv[-4]), int(sys.argv[-3]), int(sys.argv[-2]), int(sys.argv[-1]))
