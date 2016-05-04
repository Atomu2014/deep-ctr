import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from FM import FM
from LR import LR


def collect(fin, size=100000):
    buf = []
    for i in range(size):
        try:
            line = next(fin)
            buf.append(line)
        except StopIteration as e:
            break
    np.random.shuffle(buf)
    return buf


def load_ipinyou_data(path):
    fin = open(path)

    X_ind = []
    X_val = []
    y = []
    max_fea = 0
    max_dim = 0
    while True:
        buf = collect(fin)
        if len(buf) < 1:
            break

        for line in buf:
            fields = line.strip().split()
            max_fea = max(max_fea, len(fields) - 2)
            y.append(int(fields[0]))
            x_ind = [int(x.split(':')[0]) for x in fields[2:]]
            max_dim = max(max_dim, max(x_ind))
            l = len(x_ind)
            x_val = [1] * l
            X_ind.append(x_ind)
            X_val.append(x_val)

    return X_ind, X_val, y, max_dim, max_fea


def feed_zero(X_ind, X_val, y, max_dim, max_fea):
    for i in range(len(y)):
        l = len(X_ind[i])
        X_ind[i].extend([max_dim] * (max_fea - l))
        X_val[i].extend([0] * (max_fea - l))

    X_ind = np.array(X_ind)
    X_val = np.array(X_val)
    y = np.array(y)

    inds = range(len(y))
    np.random.shuffle(inds)
    X_ind = X_ind[inds]
    X_val = X_val[inds]
    y = y[inds]

    print X_ind.shape
    print X_val.shape
    print y.shape
    print X_ind.max()

    return X_ind, X_val, y


if __name__ == '__main__':
    cam = 2261
    X_train_ind, X_train_val, y_train, X_dim_train, X_feas_train = load_ipinyou_data(
        '../data/ipinyou-data/%d/train.yzx.txt' % cam)
    X_test_ind, X_test_val, y_test, X_dim_test, X_feas_test = load_ipinyou_data(
        '../data/ipinyou-data/%d/test.yzx.txt' % cam)

    print 'positive ratio', 1.0 * np.count_nonzero(y_train) / len(y_train)

    X_dim = max(X_dim_train, X_dim_test) + 2
    X_feas = max(X_feas_train, X_feas_test)

    X_train_ind, X_train_val, y_train = feed_zero(X_train_ind, X_train_val, y_train, X_dim - 1, X_feas)
    X_test_ind, X_test_val, y_test = feed_zero(X_test_ind, X_test_val, y_test, X_dim - 1, X_feas)

    eval_size = 20000
    X_test_ind = X_test_ind[:eval_size, :].reshape((eval_size * X_test_ind.shape[1]))
    X_test_val = X_test_val[:eval_size, :].reshape((eval_size * X_test_val.shape[1]))
    y_test = y_test[:eval_size]

    batch_size = 1
    algo = 'FM'
    print cam, batch_size, eval_size, algo

    if 'LR' in algo:
        epoch = 10000
        model = LR(batch_size, [X_dim, X_feas], ['uniform', -0.001, 0.001, [0x89AB], None], ['sgd', 1e-3], [1e-4],
                   'train', eval_size, X_test_ind, X_test_val)
    elif 'FM' in algo:
        epoch = 10000
        model = FM(batch_size, [X_dim, X_feas, 10], ['uniform', -0.001, 0.001, [0x3210, 0x7654], None], ['sgd', 1e-3],
                   [1e-3], 'train', eval_size, X_test_ind, X_test_val)

    print model.log
    with tf.Session(graph=model.graph) as sess:
        tf.initialize_all_variables().run()
        print 'model initialized'
        start_time = time.time()
        step = 0
        batch_preds = []
        batch_labels = []
        err_rcds = []
        it = 0
        while True:
            it += 1
            print 'iteration %d' % it
            for step in range(X_train_ind.shape[0] / batch_size):
                labels = y_train[step * batch_size:(step + 1) * batch_size]
                ids = X_train_ind[step * batch_size:(step + 1) * batch_size, :]
                wts = X_train_val[step * batch_size:(step + 1) * batch_size, :]
                ids = ids.reshape((X_feas * batch_size))
                wts = wts.reshape((X_feas * batch_size))

                wts2 = wts ** 2
                if 'LR' in algo:
                    feed_dict = {model.lbl_hldr: labels, model.sp_id_hldr: ids, model.sp_wt_hldr: wts}
                elif 'FM' in algo:
                    feed_dict = {model.sp_id_hldr: ids, model.sp_wt_hldr: wts, model.sp_wt2_hldr: wts2,
                                 model.lbl_hldr: labels}
                    # v, vx, v2x2, s_vx, s_v2x2, yhat = sess.run(
                    #     [model.V, model._Vx, model._V2x2, model.s_vx, model.s_v2x2, model.yhat], feed_dict=feed_dict)
                    # print 'v', v.shape
                    # print v
                    # print 'x', ids
                    # print wts
                    # print 'v dot x', vx.shape
                    # print vx
                    # print 'v square dot x square', v2x2.shape
                    # print v2x2
                    # print 'sum over vx', s_vx.shape
                    # print s_vx
                    # print 'sum over v2x2', s_v2x2.shape
                    # print s_v2x2
                    # print 'yhat', yhat.shape
                    # print yhat
                    # exit(0)

                _, l, p = sess.run([model.ptmzr, model.loss, model.train_preds], feed_dict=feed_dict)
                batch_preds.extend(_x[0] for _x in p)
                batch_labels.extend(labels)
                if step % epoch == 0:
                    print 'step: %d\ttime: %d\tloss: %g' % (step * batch_size, time.time() - start_time, l)
                    start_time = time.time()
                    eval_preds = model.eval_preds.eval()
                    try:
                        print 'train auc: %g' % roc_auc_score(batch_labels, batch_preds)
                        print 'eval auc: %g' % roc_auc_score(y_test, eval_preds)
                    except ValueError as e:
                        print 'train auc: None'
                        print 'eval auc: %g' % roc_auc_score(y_test, eval_preds)
