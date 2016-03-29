import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

sns.set_style('darkgrid')
colors = sns.color_palette()


def show_log(argv):
    log_path = '../log/%s' % argv[0]
    mode = argv[1]
    p1 = int(argv[2])
    p2 = int(argv[3])
    p3 = int(argv[4])
    p4 = int(argv[5])

    logs = np.loadtxt(log_path, delimiter='\t', skiprows=4)
    fig = plt.figure()
    ax0 = fig.add_subplot(311)
    ax1 = fig.add_subplot(312)
    ax2 = fig.add_subplot(313)

    if mode == 'all':
        step = logs[::p3, 0]
        batch_loss = logs[::p3, 1]
        eval_loss = logs[::p3, 2]
        batch_auc = logs[::p3, 3]
        eval_auc = logs[::p3, 4]
        batch_rmse = logs[::p3, 5]
        eval_rmse = logs[::p3, 6]
    elif mode == 'head':
        step = logs[:p1:p3, 0]
        batch_loss = logs[:p1:p3, 1]
        eval_loss = logs[:p1:p3, 2]
        batch_auc = logs[:p1:p3, 3]
        eval_auc = logs[:p1:p3, 4]
        batch_rmse = logs[:p1:p3, 5]
        eval_rmse = logs[:p1:p3, 6]
    elif mode == 'tail':
        step = logs[p2::p3, 0]
        batch_loss = logs[p2::p3, 1]
        eval_loss = logs[p2::p3, 2]
        batch_auc = logs[p2::p3, 3]
        eval_auc = logs[p2::p3, 4]
        batch_rmse = logs[p2::p3, 5]
        eval_rmse = logs[p2::p3, 6]
    else:
        step = logs[p1:p2:p3, 0]
        batch_loss = logs[p1:p2:p3, 1]
        eval_loss = logs[p1:p2:p3, 2]
        batch_auc = logs[p1:p2:p3, 3]
        eval_auc = logs[p1:p2:p3, 4]
        batch_rmse = logs[p1:p2:p3, 5]
        eval_rmse = logs[p1:p2:p3, 6]

    smooth_loss = np.array(eval_loss[p4 - 1:])
    smooth_auc = np.array(eval_auc[p4 - 1:])
    smooth_rmse = np.array(eval_rmse[p4 - 1:])
    for i in range(p4 - 1):
        smooth_loss += eval_loss[i:(i - p4 + 1)]
        smooth_auc += eval_auc[i:(i - p4 + 1)]
        smooth_rmse += eval_rmse[i:(i - p4 + 1)]
    smooth_loss /= p4
    smooth_auc /= p4
    smooth_rmse /= p4

    ax0.plot(step, batch_loss, label='batch-loss', color=colors[0])
    ax0.plot(step, eval_loss, label='eval-loss', color=colors[1])
    ax0.plot(step[p4 - 1:], smooth_loss, label='smoothed eval-loss', color=colors[2])
    ax1.plot(step, batch_auc, label='batch-auc', color=colors[3])
    ax1.plot(step, eval_auc, label='eval-auc', color=colors[4])
    ax1.plot(step[p4 - 1:], smooth_auc, label='smoothed eval-auc', color=colors[5])
    ax2.plot(step, batch_rmse, label='batch-rmse', color=colors[0])
    ax2.plot(step, eval_rmse, label='eval-rmse', color=colors[1])
    ax2.plot(step[p4 - 1:], smooth_rmse, label='smoothed eval-rmse', color=colors[2])

    ax0.legend()
    ax0.set_title(log_path)
    ax0.set_xlabel('step')
    ax1.legend()
    ax1.set_title(log_path)
    ax1.set_xlabel('step')
    ax2.legend()
    ax2.set_title(log_path)
    ax2.set_xlabel('step')
    fig.canvas.draw()
    plt.show()
    return smooth_auc


def smoothed_error(log_path, p3, p4, p5):
    logs = np.loadtxt(log_path, delimiter='\t', skiprows=3)
    step = logs[::p3, 0]
    eval_auc = logs[::p3, 4]
    smooth_auc = np.array(eval_auc[p4 - 1:])
    for i in range(p4 - 1):
        smooth_auc += eval_auc[i:(i - p4 + 1)]
    smooth_auc /= p4
    plt.plot(step[p4-1:], smooth_auc, color=colors[2])
    smooth_auc_error = smooth_auc[p5 - 1:] - smooth_auc[:1 - p5]
    plt.plot(step[p4 + p5 - 2:], smooth_auc_error, color=colors[0])
    inds = np.where(smooth_auc_error < 0)[0]
    plt.scatter(np.array(step[p4 + p5 - 2:])[inds], smooth_auc_error[inds], color=colors[1])
    plt.show()


if __name__ == '__main__':
    show_log(sys.argv[1:])
