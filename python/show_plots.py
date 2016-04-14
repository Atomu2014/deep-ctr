import cPickle as pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style('darkgrid')
colors = sns.color_palette()


def show_model(argv):
    model_path = '../model/%s' % argv[0]
    print model_path
    save = pickle.load(open(model_path, 'rb'))
    for k in sorted(save.keys()):
        v = save[k]
        if type(v) is not np.ndarray:
            print v
        elif v.ndim == 1:
            if v.shape[0] == 1:
                print v
            else:
                plt.plot(range(len(v)), v)
                plt.title(model_path)
                plt.ylabel(k)
                plt.show()
        else:
            x = np.arange(v.shape[0])
            plt.plot(x[:], v[:, ])
            plt.title(model_path)
            plt.ylabel(k)
        plt.show()


def show_log(argv):
    log_path = '../log/%s' % argv[0]
    p1 = int(argv[1])

    logs = np.loadtxt(log_path, delimiter='\t', skiprows=4, usecols=range(8))
    fig = plt.figure()
    ax0 = fig.add_subplot(311)
    ax1 = fig.add_subplot(312)
    ax2 = fig.add_subplot(313)

    step = logs[:, 0]
    batch_loss = logs[:, 1]
    eval_loss = logs[:, 2]
    batch_auc = logs[:, 3]
    eval_auc = logs[:, 4]
    batch_rmse = logs[:, 5]
    eval_rmse = logs[:, 6]

    smooth_loss = np.array(eval_loss[p1 - 1:])
    smooth_auc = np.array(eval_auc[p1 - 1:])
    smooth_rmse = np.array(eval_rmse[p1 - 1:])
    for i in range(p1 - 1):
        smooth_loss += eval_loss[i:(i - p1 + 1)]
        smooth_auc += eval_auc[i:(i - p1 + 1)]
        smooth_rmse += eval_rmse[i:(i - p1 + 1)]
    smooth_loss /= p1
    smooth_auc /= p1
    smooth_rmse /= p1

    ax0.plot(step, batch_loss, label='batch-loss', color=colors[0])
    ax0.plot(step, eval_loss, label='eval-loss', color=colors[1])
    ax0.plot(step[p1 - 1:], smooth_loss, label='smoothed eval-loss', color=colors[2])
    ax1.plot(step, batch_auc, label='batch-auc', color=colors[3])
    ax1.plot(step, eval_auc, label='eval-auc', color=colors[4])
    ax1.plot(step[p1 - 1:], smooth_auc, label='smoothed eval-auc', color=colors[5])
    ax2.plot(step, batch_rmse, label='batch-rmse', color=colors[0])
    ax2.plot(step, eval_rmse, label='eval-rmse', color=colors[1])
    ax2.plot(step[p1 - 1:], smooth_rmse, label='smoothed eval-rmse', color=colors[2])

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


def early_stop(argv):
    log_path = '../log/%s' % argv[0]
    p3 = int(argv[1])
    p4 = int(argv[2])
    p5 = int(argv[3])
    logs = np.loadtxt(log_path, delimiter='\t', skiprows=4)
    step = logs[::p3, 0]
    eval_auc = logs[::p3, 4]
    smooth_auc = np.array(eval_auc[p4 - 1:])
    for i in range(p4 - 1):
        smooth_auc += eval_auc[i:(i - p4 + 1)]
    smooth_auc /= p4
    plt.plot(step, eval_auc, color=colors[3])
    plt.plot(step[p4 - 1:], smooth_auc, color=colors[2])
    smooth_auc_error = smooth_auc[p5 - 1:] - smooth_auc[:1 - p5]
    plt.plot(step[p4 + p5 - 2:], smooth_auc_error, color=colors[0])
    inds = np.where(smooth_auc_error < 0)[0]
    plt.scatter(np.array(step[p4 + p5 - 2:])[inds], smooth_auc_error[inds], color=colors[1])
    plt.scatter(np.array(step[p4 - 1:])[inds + p5 - 1], smooth_auc[inds + p5 - 1], color=colors[1])
    plt.title(log_path)
    plt.show()


if __name__ == '__main__':
    if sys.argv[1] == 'log':
        show_log(sys.argv[2:])
    elif sys.argv[1] == 'model':
        show_model(sys.argv[2:])
    elif sys.argv[1] == 'stop':
        early_stop(sys.argv[2:])
