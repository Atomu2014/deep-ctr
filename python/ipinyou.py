import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, mean_squared_error

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


def stat(path):
    fin = open(path)

    max_fea = 0
    max_dim = 0
    while True:
        buf = collect(fin)
        if len(buf) < 1:
            break

        for line in buf:
            fields = line.strip().split()
            max_fea = max(max_fea, len(fields) - 2)
            x_ind = [int(x.split(':')[0]) for x in fields[2:]]
            max_dim = max(max_dim, max(x_ind))

    return max_dim, max_fea


def load_ipinyou_data(fin, size, max_dim, max_fea):
    X_ind = []
    X_val = []
    y = []
    buf = collect(fin, size)
    if len(buf) < 1:
        return None, None, None

    for line in buf:
        fields = line.strip().split()
        y.append(int(fields[0]))
        x_ind = [int(x.split(':')[0]) for x in fields[2:]]
        l = len(x_ind)
        x_val = [1] * l
        X_ind.append(x_ind)
        X_val.append(x_val)
        X_ind[-1].extend([max_dim] * (max_fea - l))
        X_val[-1].extend([0] * (max_fea - l))

    X_ind = np.array(X_ind)
    X_val = np.array(X_val)
    y = np.array(y)

    return X_ind, X_val, y


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


def write_log(_line, echo=False):
    with open(log_path, 'a') as log_in:
        log_in.write(_line + '\n')
        if echo:
            print _line


def watch_train(step, batch_loss, batch_labels, batch_preds, eval_preds, eval_labels):
    try:
        batch_auc = np.float32(roc_auc_score(batch_labels, batch_preds))
    except ValueError:
        batch_auc = -1
    try:
        eval_auc = np.float32(roc_auc_score(eval_labels, eval_preds))
    except ValueError:
        eval_auc = -1
    log = '%d\t%g\t%g\t%g\t' % (step, batch_auc, eval_auc, batch_loss)
    write_log(log)
    print 'train auc: %g\teval auc: %g' % (batch_auc, eval_auc)


if __name__ == '__main__':
    cam = 'all'
    train_path = '../data/ipinyou-data/%s/train.yzx.txt.shuf' % cam
    test_path = '../data/ipinyou-data/%s/test.yzx.txt.shuf' % cam

    X_dim_train, X_feas_train = stat(train_path)
    X_dim_test, X_feas_test = stat(test_path)
    X_dim = max(X_dim_train, X_dim_test) + 2
    X_feas = max(X_feas_train, X_feas_test)

    algo = 'FM'

    tag = (str(cam) + ' ' + time.strftime('%c') + ' ' + algo).replace(' ', '_')
    log_path = '../log/%s' % tag
    print log_path

    if 'LR' in algo:
        batch_size = 1
        epoch = 100000
        eval_size = 100000
        model = LR(batch_size, [X_dim, X_feas], ['uniform', -0.001, 0.001, [0x89AB], None], ['sgd', 1e-3], [1e-3],
                   'train', eval_size)
    elif 'FM' in algo:
        batch_size = 1
        epoch = 10000
        eval_size = 100000
        model = FM(batch_size, [X_dim, X_feas, 10], ['uniform', -0.001, 0.001, [0x3210, 0x7654], None], ['sgd', 1e-3],
                   [1e-2], 'train', eval_size)

    print batch_size, epoch, eval_size

    write_log(model.log, True)

    with tf.Session(graph=model.graph) as sess:
        tf.initialize_all_variables().run()
        print 'model initialized'

        it = 0
        while True:
            it += 1
            print 'iteration %d' % it

            train_data_set = open(train_path, 'rb')
            start_time = time.time()
            step = 0

            while True:
                batch_ids, batch_wts, batch_labels = load_ipinyou_data(train_data_set, epoch, X_dim - 1, X_feas)

                if batch_ids is None:
                    break

                wts2 = batch_wts ** 2
                batch_preds = []
                for _i in range(len(batch_labels) / batch_size):
                    feed_dict = {model.lbl_hldr: batch_labels[_i * batch_size: (_i + 1) * batch_size],
                                 model.sp_id_hldr: batch_ids[_i * batch_size: (_i + 1) * batch_size].flatten(),
                                 model.sp_wt_hldr: batch_wts[_i * batch_size: (_i + 1) * batch_size].flatten()}

                    _, l, p = sess.run([model.ptmzr, model.loss, model.train_preds], feed_dict=feed_dict)
                    batch_preds.extend(_x[0] for _x in p)

                step += len(batch_ids)
                if step % epoch == 0:
                    print 'step: %d\ttime: %d\tloss: %g' % (step * batch_size, time.time() - start_time, l)
                    test_data_set = open(test_path, 'rb')
                    start_time = time.time()
                    eval_preds = []
                    eval_labels = []

                    while True:
                        _ids, _wts, _labels = load_ipinyou_data(test_data_set, eval_size, X_dim - 1, X_feas)

                        if _ids is None or _ids.shape[0] < eval_size:
                            break

                        _ids = _ids.flatten()
                        _wts = _wts.flatten()

                        feed_dict = {model.eval_id_hldr: _ids, model.eval_wt_hldr: _wts}
                        eval_preds.extend(model.eval_preds.eval(feed_dict=feed_dict))
                        eval_labels.extend(_labels)

                        if step % (10 * epoch) != 0 and len(eval_labels) >= 10 * eval_size:
                            break

                    watch_train(step, l, batch_labels, batch_preds, eval_preds, eval_labels)
