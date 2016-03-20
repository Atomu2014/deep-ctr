import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import roc_auc_score

sns.set_style('darkgrid')

# scale valued-fields to [0, 1]
max_vals = [65535, 8000, 705, 249000, 7974, 14159, 946, 43970, 5448, 11, 471, 35070017, 8376]
cat_sizes = np.array(
    [631839, 25297, 15534, 7109, 19273, 4, 6754, 1303, 55, 527815, 147691, 110674, 11, 2202, 9023, 63, 5, 946, 15,
     639542, 357567, 600328, 118442, 10217, 78, 34])
# brute feature selection
mask = np.where(cat_sizes > 0)[0]
offsets = [13 + sum(cat_sizes[mask[:i]]) for i in range(len(mask))]
X_dim = 13 + np.sum(cat_sizes[mask])
# num of fields
X_feas = 13 + len(mask)

print len(mask), np.sum(cat_sizes[mask])

fin = None
file_list = ['../data/nds.10.shuf.ind', '../data/nds.10.shuf.ind']
file_index = 0
line_index = 0

batch_size = 1
buffer_size = 100000
eval_size = 1000
epoch = 100

# 'LR', 'FM'
_algo = 'LR'
_rank = 2
_log_path = '../log/' + time.strftime('%c') + ' ' + _algo
if _algo == 'FM':
    _log_path += str(_rank)

_learning_rate = 0.0001
# _alpha = 1
_lambda = 0.01
_keep_prob = 0.5
# 'normal', 't-normal', 'uniform'(default)
_init_method = 'uniform'
_stddev = 0.001
_min_val = -0.001
_max_val = -1 * _min_val

_nds_rate = 0.1


# extract feature from a line
def get_fxy(_line):
    fields = _line.split('\t')
    _y = int(fields[0])
    cats = fields[14:]
    _f = range(13)
    _f.extend([int(cats[mask[_i]]) + offsets[_i] for _i in range(len(mask))])
    _x = [float(fields[_i]) / max_vals[_i - 1] for _i in range(1, 14)]
    _x.extend([1] * len(mask))
    return _y, _f, _x


# generate a batch of data
def get_batch_sparse_tensor(file_name, start_index, size, row_start=0):
    global fin
    if fin is None:
        fin = open(file_name, 'r')
    labels = []
    cols = []
    values = []
    rows = []
    row_num = row_start
    for _i in range(start_index, start_index + size):
        try:
            _line = next(fin)
            if len(_line.strip()):
                _y, _f, _x = get_fxy(_line)
                rows.extend([row_num] * len(_f))
                cols.extend(_f)
                values.extend(_x)
                labels.append(_y)
                row_num += 1
            else:
                break
        except StopIteration as e:
            print e
            fin = None
            break

    return labels, rows, cols, values


def get_batch_xy(size):
    global file_index, line_index
    labels, rows, cols, values = get_batch_sparse_tensor(file_list[file_index], line_index, size)
    if len(labels) == size:
        line_index += size
        return np.array(labels), np.array(rows), np.array(cols), np.array(values)

    file_index = (file_index + 1) % len(file_list)
    line_index = size - len(labels)
    l, r, c, v = get_batch_sparse_tensor(file_list[file_index], 0, size - len(labels))
    labels.extend(l)
    rows.extend(r)
    cols.extend(c)
    values.extend(v)
    return np.array(labels), np.array(rows), np.array(cols), np.array(values)


def write_log(vals):
    with open(_log_path, 'a') as log_in:
        log_in.write('\t'.join([str(_x) for _x in vals]) + '\n')


plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)


def show_log():
    logs = np.loadtxt(_log_path, delimiter='\t', skiprows=3)
    if logs.ndim < 2:
        return
    ax.clear()
    ax.plot(logs[-10000:, 0], logs[-10000:, 1], label='loss')
    ax.plot(logs[-10000:, 0], logs[-10000:, 2], label='batch-auc')
    ax.plot(logs[-10000:, 0], logs[-10000:, 3], label='eval-auc')
    ax.legend()
    ax.set_title(_log_path)
    ax.set_xlabel('step')
    fig.canvas.draw()


print _log_path

headers = ['batch: %d, epoch: %d, buffer_size: %d, eval_size: %d' % (batch_size, epoch, buffer_size, eval_size),
           'lr: %f, lambda: %f, keep_prob: %f' % (_learning_rate, _lambda, _keep_prob),
           'init method: %s, stddev: %f, interval: [%f, %f]' % (_init_method, _stddev, _min_val, _max_val)]

for h in headers:
    write_log([h])
    print h

sp_train_inds = []
sp_nds_valid_inds = []
sp_valid_inds = []

valid_labels = []
valid_cols = []
valid_vals = []
valid_num_row = 0

with open('../data/test.unif.10.shuf.ind', 'r') as valid_fin:
    buf = []
    for line in valid_fin:
        buf.append(line)
        valid_num_row += 1
        if valid_num_row == eval_size:
            break
    for line in buf:
        y, f, x = get_fxy(line)
        valid_cols.extend(f)
        valid_vals.extend(x)
        valid_labels.append(y)

for i in range(valid_num_row):
    for j in range(X_feas):
        sp_valid_inds.append([i, j])
valid_vals2 = np.array(valid_vals) ** 2

for i in range(batch_size):
    for j in range(X_feas):
        sp_train_inds.append([i, j])


def lr():
    graph = tf.Graph()
    with graph.as_default():
        tf_sp_id_vals = tf.placeholder(tf.int64, shape=[batch_size * X_feas])
        tf_sp_weight_vals = tf.placeholder(tf.float32, shape=[batch_size * X_feas])
        tf_sp_ids = tf.SparseTensor(sp_train_inds, tf_sp_id_vals, shape=[batch_size, X_feas])
        tf_sp_weights = tf.SparseTensor(sp_train_inds, tf_sp_weight_vals, shape=[batch_size, X_feas])
        tf_train_label = tf.placeholder(tf.float32)
        tf_sp_valid_ids = tf.SparseTensor(sp_valid_inds, valid_cols, shape=[valid_num_row, X_feas])
        tf_sp_valid_weights = tf.SparseTensor(sp_valid_inds, valid_vals, shape=[valid_num_row, X_feas])

        if _init_method == 'normal':
            weights = tf.Variable(tf.random_normal([X_dim, 1], stddev=_stddev))
        elif _init_method == 't-normal':
            weights = tf.Variable(tf.truncated_normal([X_dim, 1], stddev=_stddev))
        else:
            weights = tf.Variable(tf.random_uniform([X_dim, 1], minval=_min_val, maxval=_max_val))
        bias = tf.Variable(0.0)

        logits = tf.nn.embedding_lookup_sparse(weights, tf_sp_ids, tf_sp_weights, combiner='sum') + bias
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, tf_train_label)) + _lambda * (
            tf.nn.l2_loss(weights) + tf.nn.l2_loss(bias))
        optimizer = tf.train.GradientDescentOptimizer(_learning_rate).minimize(loss)

        train_pred = tf.sigmoid(logits)
        unif_valid_logits = tf.nn.embedding_lookup_sparse(weights, tf_sp_valid_ids, tf_sp_valid_weights,
                                                          combiner='sum')
        valid_pred = tf.sigmoid(unif_valid_logits)
        valid_pred /= valid_pred + (1 - valid_pred) / _nds_rate

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print 'initialized'
        step = 0
        batch_preds = []
        batch_labels = []
        start_time = time.time()
        while True:
            label, _, cols, values = get_batch_xy(buffer_size)
            for _i in range(label.shape[0] / batch_size):
                step += 1
                feed_dict = {tf_sp_id_vals: cols[_i * batch_size * X_feas: (_i + 1) * batch_size * X_feas],
                             tf_sp_weight_vals: values[_i * batch_size * X_feas: (_i + 1) * batch_size * X_feas],
                             tf_train_label: label[_i * batch_size: (_i + 1) * batch_size]}
                _, l, pred = session.run([optimizer, loss, train_pred], feed_dict=feed_dict)

                batch_preds.extend(_x[0] for _x in pred)
                batch_labels.extend(label[_i * batch_size: (_i + 1) * batch_size])

                if step % epoch == 0:
                    print 'loss as step %d: %f\ttime: %d' % (step, l, time.time() - start_time)
                    start_time = time.time()
                    try:
                        batch_auc = roc_auc_score(batch_labels, batch_preds)
                        batch_preds = []
                        batch_labels = []
                        valid_auc = roc_auc_score(valid_labels, valid_pred.eval())
                        print 'batch-auc:%.4f\teval-auc: %.4f' % (batch_auc, valid_auc)
                    except ValueError:
                        batch_auc = 0
                        valid_auc = roc_auc_score(valid_labels, valid_pred.eval())
                        print 'batch-auc:None\teval-auc: %.4f' % valid_auc
                    write_log([step, l, batch_auc, valid_auc])
                    show_log()


def fm():
    print 'rank: %d' % _rank
    graph = tf.Graph()

    def factorization(sp_ids, sp_weights, sp_weights2):
        yhat = tf.nn.embedding_lookup_sparse(W, sp_ids, sp_weights, combiner='sum') + b
        _Vx = tf.nn.embedding_lookup_sparse(V, sp_ids, sp_weights, combiner='sum')
        _V2x2 = tf.nn.embedding_lookup_sparse(tf.square(V), sp_ids, sp_weights2, combiner='sum')
        yhat += 0.5 * tf.reshape(tf.reduce_sum(tf.matmul(_Vx, _Vx, transpose_b=True), 1) - tf.reduce_sum(_V2x2, 1),
                                 shape=[-1, 1])
        return yhat

    with graph.as_default():
        tf_sp_id_vals = tf.placeholder(tf.int64, shape=[batch_size * X_feas])
        tf_sp_weight_vals = tf.placeholder(tf.float32, shape=[batch_size * X_feas])
        tf_sp_ids = tf.SparseTensor(sp_train_inds, tf_sp_id_vals, shape=[batch_size, X_feas])
        tf_sp_weights = tf.SparseTensor(sp_train_inds, tf_sp_weight_vals, shape=[batch_size, X_feas])
        tf_train_label = tf.placeholder(tf.float32)
        tf_sp_weight2_vals = tf.placeholder(tf.float32, shape=[batch_size * X_feas])
        tf_sp_weights2 = tf.SparseTensor(sp_train_inds, tf_sp_weight2_vals, shape=[batch_size, X_feas])
        tf_sp_valid_ids = tf.SparseTensor(sp_valid_inds, valid_cols, shape=[valid_num_row, X_feas])
        tf_sp_valid_weights = tf.SparseTensor(sp_valid_inds, valid_vals, shape=[valid_num_row, X_feas])
        tf_sp_valid_weights2 = tf.SparseTensor(sp_valid_inds, valid_vals2, shape=[valid_num_row, X_feas])

        if _init_method == 'normal':
            W = tf.Variable(tf.random_normal([X_dim, 1], stddev=_stddev))
            V = tf.Variable(tf.random_normal([X_dim, _rank], stddev=_stddev))
        elif _init_method == 't-normal':
            W = tf.Variable(tf.truncated_normal([X_dim, 1], stddev=_stddev))
            V = tf.Variable(tf.truncated_normal([X_dim, _rank], stddev=_stddev))
        else:
            W = tf.Variable(tf.random_uniform([X_dim, 1], minval=_min_val, maxval=_max_val))
            V = tf.Variable(tf.random_uniform([X_dim, _rank], minval=_min_val, maxval=_max_val))
        b = tf.Variable(0.0)

        logit = factorization(tf_sp_ids, tf_sp_weights, tf_sp_weights2)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logit, tf_train_label)) + _lambda * (
            tf.nn.l2_loss(W) + tf.nn.l2_loss(V) + tf.nn.l2_loss(b))
        optimizer = tf.train.GradientDescentOptimizer(_learning_rate).minimize(loss)
        train_pred = tf.sigmoid(logit)
        valid_logits = factorization(tf_sp_valid_ids, tf_sp_valid_weights, tf_sp_valid_weights2)
        valid_pred = tf.sigmoid(valid_logits)
        valid_pred /= valid_pred + (1 - valid_pred) / _nds_rate

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print 'initialized'
        step = 0
        batch_preds = []
        batch_labels = []
        start_time = time.time()
        while True:
            label, rows, cols, values = get_batch_xy(buffer_size)
            for _i in range(label.shape[0] / batch_size):
                step += 1
                _label = label[_i * batch_size: (_i + 1) * batch_size]
                _cols = cols[_i * batch_size * X_feas: (_i + 1) * batch_size * X_feas]
                _vals = values[_i * batch_size * X_feas: (_i + 1) * batch_size * X_feas]
                _vals2 = _vals ** 2

                feed_dict = {tf_sp_id_vals: _cols, tf_sp_weight_vals: _vals, tf_sp_weight2_vals: _vals2,
                             tf_train_label: _label}
                _, l, pred, lg = session.run([optimizer, loss, train_pred, logit], feed_dict=feed_dict)
                batch_preds.extend(_x[0] for _x in pred)
                batch_labels.extend(label[_i * batch_size: (_i + 1) * batch_size])
                if step % epoch == 0:
                    print 'loss as step %d: %f\ttime: %d' % (step, l, time.time() - start_time)
                    start_time = time.time()
                    try:
                        batch_auc = roc_auc_score(batch_labels, batch_preds)
                        batch_preds = []
                        batch_labels = []
                        valid_auc = roc_auc_score(valid_labels, valid_pred.eval())
                        print 'batch-auc:%.4f\teval-auc: %.4f' % (batch_auc, valid_auc)
                    except ValueError:
                        batch_auc = 0
                        valid_auc = roc_auc_score(valid_labels, valid_pred.eval())
                        print 'batch-auc:None\teval-auc: %.4f' % valid_auc
                    write_log([step, l, batch_auc, valid_auc])
                    show_log()


if __name__ == '__main__':
    if _algo == 'LR':
        lr()
    elif _algo == 'FM':
        fm()
