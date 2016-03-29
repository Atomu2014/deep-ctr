import time

# import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import tensorflow as tf
from sklearn.metrics import roc_auc_score, mean_squared_error, log_loss

# sns.set_style('darkgrid')

# scale valued-fields to [0, 1]
max_vals = [65535, 8000, 705, 249000, 7974, 14159, 946, 43970, 5448, 11, 471, 35070017, 8376]
cat_sizes = np.array([631839, 25297, 15534, 7109, 19273, 4, 6754, 1303, 55, 527815, 147691, 110674,
                      11, 2202, 9023, 63, 5, 946, 15, 639542, 357567, 600328, 118442, 10217, 78, 34])

# brute feature selection
mask = np.where(cat_sizes > 0)[0]
offsets = [13 + sum(cat_sizes[mask[:i]]) for i in range(len(mask))]
X_dim = 13 + np.sum(cat_sizes[mask])
# num of fields
X_feas = 13 + len(mask)

print len(mask), np.sum(cat_sizes[mask])

train_path = '../data/nds.10.shuf.ind'
eval_path = '../data/test.nds.10.shuf.ind'
fin = None

batch_size = 10
buffer_size = 1000000
eval_size = 10000
epoch = 1000
nds_rate = 0.1

# 'create', 'restore'
train_mode = 'create'
ckpt = epoch * 10
# 'LR', 'FM'
algo = 'LR'
rank = 2
if train_mode == 'create':
    tag = (time.strftime('%c') + ' ' + algo).replace(' ', '_')
    if algo == 'FM':
        tag += str(rank)
else:
    tag = ''
log_path = '../log/%s' % tag
model_path = '../model/%s' % tag

print train_mode, log_path

_learning_rate = 0.00001
# _alpha = 1
_lambda = 0.005
_epsilon = 1e-8
_keep_prob = 0.5
# 'normal', 't-normal', 'uniform'(default)
_init_method = 'uniform'
_stddev = 0.001
_min_val = -0.001
_max_val = -1 * _min_val
_seeds = [0x01234567, 0x89ABCDEF]

least_step = 100 * epoch
skip_window = 10
smooth_window = 10
stop_window = 10


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
def get_batch_sparse_tensor(file_name, size, row_start=0):
    global fin
    if fin is None:
        fin = open(file_name, 'r')
    labels = []
    cols = []
    values = []
    rows = []
    row_num = row_start
    for _i in range(size):
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
    labels, rows, cols, values = get_batch_sparse_tensor(train_path, size)
    if len(labels) == size:
        return np.array(labels), np.array(rows), np.array(cols), np.array(values)

    l, r, c, v = get_batch_sparse_tensor(train_path, size - len(labels))
    labels.extend(l)
    rows.extend(r)
    cols.extend(c)
    values.extend(v)
    return np.array(labels), np.array(rows), np.array(cols), np.array(values)


def write_log(vals, erase=False):
    if erase:
        mode = 'w'
    else:
        mode = 'a'
    with open(log_path, mode) as log_in:
        log_in.write('\t'.join([str(_x) for _x in vals]) + '\n')


headers = ['train: %s, eval: %s' % (train_path, eval_path),
           'batch: %d, epoch: %d, buffer_size: %d, eval_size: %d, checkpoint: %d, skip: %d, smooth: %d, stop: %d' % (
               batch_size, epoch, buffer_size, eval_size, ckpt, skip_window, smooth_window, stop_window),
           'lr: %f, lambda: %f, epsilon:%f, keep_prob: %f' % (_learning_rate, _lambda, _epsilon, _keep_prob),
           'init method: %s, stddev: %f, interval: [%f, %f], seeds: %s' % (_init_method, _stddev, _min_val, _max_val, str(_seeds))]

for h in headers:
    write_log([h], h == headers[0])
    print h

sp_train_inds = []
sp_nds_valid_inds = []
sp_valid_inds = []

valid_labels = []
valid_cols = []
valid_vals = []
valid_num_row = 0

with open(eval_path, 'r') as valid_fin:
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


def pre_train(sess, var_map):
    saver = tf.train.Saver(var_map)
    if train_mode == 'create':
        tf.initialize_all_variables().run()
        print 'model initialized'
    else:
        saver.restore(sess, model_path)
        print 'model restored'
    return saver


def watch_train(sess, saver, step, batch_loss, start_time, batch_labels, batch_preds, valid_preds, auc_track):
    if step % epoch == 0:
        print 'step: %d\ttime: %d' % (step, time.time() - start_time)
        try:
            batch_auc = roc_auc_score(batch_labels, batch_preds)
        except ValueError:
            batch_auc = -1
        valid_auc = roc_auc_score(valid_labels, valid_preds)
        auc_track.append(valid_auc)
        auc_track = auc_track[2 * skip_window * (stop_window + smooth_window):]
        batch_rmse = np.sqrt(mean_squared_error(batch_labels, batch_preds))
        valid_rmse = np.sqrt(mean_squared_error(valid_labels, valid_preds))
        valid_loss = log_loss(valid_labels, valid_preds)
        print 'batch-loss: %.4f\teval-loss: %.4f\tbatch-auc:%.4f\teval-auc: %.4f\tbatch-rmse: %f\teval-rmse: %f' % (batch_loss, valid_loss, batch_auc, valid_auc, batch_rmse, valid_rmse)
        write_log([step, batch_loss, valid_loss, batch_auc, valid_auc, batch_rmse, valid_rmse])
        if step % ckpt == 0:
            saver.save(sess, model_path + '_%d' % step)
            print 'model saved as: %s_%d' % (model_path, step)
        return True
    return False


def early_stop(step, eval_auc):
    if step > least_step:
        skip_auc = eval_auc[::skip_window]
        smooth_auc = np.array(skip_auc[smooth_window - 1:])
        for i in range(smooth_auc - 1):
            smooth_auc += skip_auc[i:(i - p4 + 1)]
        smooth_auc /= smooth_window
        smooth_error = smooth_auc[stop_window - 1:] - smooth_auc[:1 - stop_window]
        if smooth_error[-1] < 0:
            print 'early stop at step %d' % step
            print 'smoothed error', str(smooth_error)
            return True
        return False
    return False


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
            weights = tf.Variable(tf.random_normal([X_dim, 1], stddev=_stddev, seed=_seeds[0]))
        elif _init_method == 't-normal':
            weights = tf.Variable(tf.truncated_normal([X_dim, 1], stddev=_stddev, seed=_seeds[0]))
        else:
            weights = tf.Variable(tf.random_uniform([X_dim, 1], minval=_min_val, maxval=_max_val, seed=_seeds[0]))
        bias = tf.Variable(0.0)

        logits = tf.nn.embedding_lookup_sparse(weights, tf_sp_ids, tf_sp_weights, combiner='sum') + bias
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, tf_train_label)) + _lambda * (
            tf.nn.l2_loss(weights) + tf.nn.l2_loss(bias))
        # optimizer = tf.train.GradientDescentOptimizer(_learning_rate).minimize(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=_learning_rate, epsilon=_epsilon).minimize(loss)

        train_pred = tf.sigmoid(logits)
        unif_valid_logits = tf.nn.embedding_lookup_sparse(weights, tf_sp_valid_ids, tf_sp_valid_weights,
                                                          combiner='sum')
        valid_preds = tf.sigmoid(unif_valid_logits)
        valid_preds /= valid_preds + (1 - valid_preds) / nds_rate

    with tf.Session(graph=graph) as sess:
        saver = pre_train(sess, {'weights': weights, 'bias': bias})
        step = 0
        batch_preds = []
        batch_labels = []
        start_time = time.time()
        auc_track = []
        while True:
            label, _, cols, values = get_batch_xy(buffer_size)
            for _i in range(label.shape[0] / batch_size):
                step += 1
                feed_dict = {tf_sp_id_vals: cols[_i * batch_size * X_feas: (_i + 1) * batch_size * X_feas],
                             tf_sp_weight_vals: values[_i * batch_size * X_feas: (_i + 1) * batch_size * X_feas],
                             tf_train_label: label[_i * batch_size: (_i + 1) * batch_size]}
                _, l, pred = sess.run([optimizer, loss, train_pred], feed_dict=feed_dict)

                batch_preds.extend(_x[0] for _x in pred)
                batch_labels.extend(label[_i * batch_size: (_i + 1) * batch_size])
                if watch_train(sess, saver, step, l, start_time, batch_labels, batch_preds, valid_preds.eval(), auc_track):
                    start_time = time.time()
                    batch_preds = []
                    batch_labels = []
                    if early_stop(step, auc_track):
                        return


def fm():
    print 'rank: %d' % rank
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
            W = tf.Variable(tf.random_normal([X_dim, 1], stddev=_stddev, seed=_seeds[0]))
            V = tf.Variable(tf.random_normal([X_dim, rank], stddev=_stddev, seed=_seeds[1]))
        elif _init_method == 't-normal':
            W = tf.Variable(tf.truncated_normal([X_dim, 1], stddev=_stddev, seed=_seeds[0]))
            V = tf.Variable(tf.truncated_normal([X_dim, rank], stddev=_stddev, seed=_seeds[1]))
        else:
            W = tf.Variable(tf.random_uniform([X_dim, 1], minval=_min_val, maxval=_max_val, seed=_seeds[0]))
            V = tf.Variable(tf.random_uniform([X_dim, rank], minval=_min_val, maxval=_max_val, seed=_seeds[1]))
        b = tf.Variable(0.0)

        logit = factorization(tf_sp_ids, tf_sp_weights, tf_sp_weights2)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logit, tf_train_label)) + _lambda * (
            tf.nn.l2_loss(W) + tf.nn.l2_loss(V) + tf.nn.l2_loss(b))
        # optimizer = tf.train.GradientDescentOptimizer(_learning_rate).minimize(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=_learning_rate, epsilon=_epsilon).minimize(loss)
        train_pred = tf.sigmoid(logit)
        valid_logits = factorization(tf_sp_valid_ids, tf_sp_valid_weights, tf_sp_valid_weights2)
        valid_pred = tf.sigmoid(valid_logits)
        valid_pred /= valid_pred + (1 - valid_pred) / nds_rate

    with tf.Session(graph=graph) as sess:
        saver = pre_train(sess, {'W': W, 'V': V, 'b': b})
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
                _, l, pred, lg = sess.run([optimizer, loss, train_pred, logit], feed_dict=feed_dict)
                batch_preds.extend(_x[0] for _x in pred)
                batch_labels.extend(label[_i * batch_size: (_i + 1) * batch_size])
                if watch_train(sess, saver, step, l, start_time, batch_labels, batch_preds, valid_pred.eval(), auc_track):
                    start_time = time.time()
                    batch_preds = []
                    batch_labels = []
                    if early_stop(step, auc_track):
                        return


if __name__ == '__main__':
    if algo == 'LR':
        lr()
    elif algo == 'FM':
        fm()
