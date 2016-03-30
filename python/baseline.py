import time

import numpy as np
import cPickle as pickle
import tensorflow as tf
from sklearn.metrics import roc_auc_score, mean_squared_error, log_loss


train_path = '../data/nds.2.5.shuf.ind.10'
eval_path = '../data/test.nds.2.5.shuf.ind.10'
nds_rate = 0.025
re_calibration = 'nds' not in '../data/test.nds.2.5.shuf.ind.10'
fin = None

max_vals = [65535, 8000, 715, 376613, 7995, 1430, 2191, 25462, 5337, 8, 221, 3023897, 8072]
cat_sizes = np.array([284414, 22289, 15249, 7022, 18956, 3, 6730, 1286, 50, 253234, 93011, 81371,
                      10, 2187, 8396, 61, 4, 932, 14, 286828, 190408, 274540, 79781, 9690, 70, 33])
cat_sizes += 1
# brute feature selection
offsets = [13 + sum(cat_sizes[:i]) for i in range(len(cat_sizes))]
X_dim = 13 + np.sum(cat_sizes)
# num of fields
X_feas = 13 + len(cat_sizes)

print 'max_vals:', max_vals
print 'cat_sizes (including \'other\'):', cat_sizes
print 'dimension: %d, features: %d' % (X_dim, X_feas)

# 'LR', 'FMxxx', 'FNN'
algo = 'LR'
tag = (time.strftime('%c') + ' ' + algo).replace(' ', '_')
if algo == 'FM':
    rank = int(algo[2:])
log_path = '../log/%s' % tag
model_path = '../model/%s.pickle' % tag

print log_path, model_path

batch_size = 100
epoch = 1000
buffer_size = 1000000
eval_size = 10000
ckpt = 10 * epoch
least_step = 1000 * epoch
skip_window = 1
smooth_window = 100
stop_window = 10

_learning_rate = 0.00001
# _alpha = 1
_lambda = 0.001
_epsilon = 1e-5
_keep_prob = 0.5
# 'normal', 't-normal', 'uniform'(default)
_init_method = 'uniform'
_stddev = 0.001
_min_val = -0.001
_max_val = -1 * _min_val
_seeds = [0x01234567, 0x89ABCDEF]

headers = ['train: %s, eval: %s, tag: %s, nds_rate: %f, re_calibration: %s' % (train_path, eval_path, tag, nds_rate, str(re_calibration)),
           'batch: %d, epoch: %d, buffer_size: %d, eval_size: %d, checkpoint: %d, least_step: %d, skip: %d, smooth: %d, stop: %d' % (
               batch_size, epoch, buffer_size, eval_size, ckpt, least_step, skip_window, smooth_window, stop_window),
           'learning_rate: %f, lambda: %f, epsilon:%f, keep_prob: %f' % (_learning_rate, _lambda, _epsilon, _keep_prob),
           'init method: %s, stddev: %f, interval: [%f, %f], seeds: %s' % (_init_method, _stddev, _min_val, _max_val, str(_seeds))]


def write_log(vals, erase=False):
    if erase:
        mode = 'w'
    else:
        mode = 'a'
    with open(log_path, mode) as log_in:
        log_in.write('\t'.join([str(_x) for _x in vals]) + '\n')

for h in headers:
    write_log([h], h == headers[0])
    print h


def get_fxy(_line):
    fields = _line.split('\t')
    _y = int(fields[0])
    cats = fields[14:]
    _f = range(13)
    _f.extend([int(cats[_i]) + offsets[_i] for _i in range(len(cat_sizes))])
    _x = [float(fields[_i]) / max_vals[_i - 1] for _i in range(1, 14)]
    _x.extend([1] * len(cat_sizes))
    return _y, _f, _x


def get_batch_sparse_tensor(size, row_start=0):
    global fin
    if fin is None:
        fin = open(train_path, 'rb')
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
    labels, rows, cols, values = get_batch_sparse_tensor(size)
    if len(labels) == size:
        return np.array(labels), np.array(rows), np.array(cols), np.array(values)

    l, r, c, v = get_batch_sparse_tensor(size - len(labels))
    labels.extend(l)
    rows.extend(r)
    cols.extend(c)
    values.extend(v)
    return np.array(labels), np.array(rows), np.array(cols), np.array(values)


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

sp_train_fld_inds = []
for i in range(batch_size):
    sp_train_fld_inds.append([i, 1])
    for j in range(X_feas):
        sp_train_inds.append([i, j])


def watch_train(sess, step, batch_loss, start_time, batch_labels, batch_preds, valid_preds, auc_track):
    if step % epoch == 0:
        print 'step: %d\ttime: %d' % (step, time.time() - start_time)
        try:
            batch_auc = roc_auc_score(batch_labels, batch_preds)
        except ValueError:
            batch_auc = -1
        valid_auc = roc_auc_score(valid_labels, valid_preds)
        auc_track.append(valid_auc)
        auc_track = auc_track[-2 * skip_window * (stop_window + smooth_window):]
        batch_rmse = np.sqrt(mean_squared_error(batch_labels, batch_preds))
        valid_rmse = np.sqrt(mean_squared_error(valid_labels, valid_preds))
        valid_loss = log_loss(valid_labels, valid_preds)
        write_log([step, batch_loss, valid_loss, batch_auc, valid_auc, batch_rmse, valid_rmse])
        print 'batch-loss: %.4f\teval-loss: %.4f\tbatch-auc:%.4f\teval-auc: %.4f\tbatch-rmse: %f\teval-rmse: %f' % (batch_loss, valid_loss, batch_auc, valid_auc, batch_rmse, valid_rmse)
        return True
    return False


def early_stop(step, eval_auc):
    if step > least_step:
        skip_auc = eval_auc[::skip_window]
        smooth_auc = np.array(skip_auc[smooth_window - 1:])
        for i in range(smooth_window - 1):
            smooth_auc += skip_auc[i:(i - smooth_window + 1)]
        smooth_auc /= smooth_window
        if len(smooth_auc) < stop_window:
            return False
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

        weights = tf.Variable(tf.random_uniform([X_dim, 1], minval=_min_val, maxval=_max_val, seed=_seeds[0]))
        bias = tf.Variable(0.0)

        logits = tf.nn.embedding_lookup_sparse(weights, tf_sp_ids, tf_sp_weights, combiner='sum') + bias
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, tf_train_label)) + _lambda * (
            tf.nn.l2_loss(weights) + tf.nn.l2_loss(bias))
        optimizer = tf.train.AdamOptimizer(learning_rate=_learning_rate, epsilon=_epsilon).minimize(loss)

        train_pred = tf.sigmoid(logits)
        unif_valid_logits = tf.nn.embedding_lookup_sparse(weights, tf_sp_valid_ids, tf_sp_valid_weights,
                                                          combiner='sum')
        valid_preds = tf.sigmoid(unif_valid_logits)
        if re_calibration:
            valid_preds /= valid_preds + (1 - valid_preds) / nds_rate

    with tf.Session(graph=graph) as sess:
        tf.initialize_all_variables().run()
        print 'model initialized'
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
                if watch_train(sess, step, l, start_time, batch_labels, batch_preds, valid_preds.eval(), auc_track):
                    start_time = time.time()
                    batch_preds = []
                    batch_labels = []
                    if step % ckpt == 0:
                        eval_map = {'weights': weights.eval(), 'bias': bias.eval()}
                        pickle.dump(eval_map, open(model_path, 'wb'))
                        print 'model dumped at %s' % model_path
                    if early_stop(step, auc_track):
                        return


def fm():
    print 'rank: %d' % rank
    graph = tf.Graph()
    valid_vals2 = np.array(valid_vals) ** 2

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

        W = tf.Variable(tf.random_uniform([X_dim, 1], minval=_min_val, maxval=_max_val, seed=_seeds[0]))
        V = tf.Variable(tf.random_uniform([X_dim, rank], minval=_min_val, maxval=_max_val, seed=_seeds[1]))
        b = tf.Variable(0.0)

        logit = factorization(tf_sp_ids, tf_sp_weights, tf_sp_weights2)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logit, tf_train_label)) + _lambda * (
            tf.nn.l2_loss(W) + tf.nn.l2_loss(V) + tf.nn.l2_loss(b))
        optimizer = tf.train.AdamOptimizer(learning_rate=_learning_rate, epsilon=_epsilon).minimize(loss)
        train_pred = tf.sigmoid(logit)
        valid_logits = factorization(tf_sp_valid_ids, tf_sp_valid_weights, tf_sp_valid_weights2)
        valid_pred = tf.sigmoid(valid_logits)
        if re_calibration:
            valid_pred /= valid_pred + (1 - valid_pred) / nds_rate

    with tf.Session(graph=graph) as sess:
        tf.initialize_all_variables().run()
        print 'model initialized'
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
                if watch_train(sess, step, l, start_time, batch_labels, batch_preds, valid_pred.eval(), auc_track):
                    start_time = time.time()
                    batch_preds = []
                    batch_labels = []
                    if step % ckpt == 0:
                        eval_map = {'W': W, 'V': V, 'b': b}
                        pickle.dump(eval_map, open(model_path, 'wb'))
                        print 'model dumped at %s' % model_path
                    if early_stop(step, auc_track):
                        return


def fnn():
    print 'fnn, rank: %d' % rank
    graph = tf.Graph()

    with graph.as_default():
        tf_fm_w = tf.Variable(tf.random_uniform([X_dim, 1], minval=_min_val, maxval=_max_val, seed=_seeds[0]))
        tf_fm_v = tf.Variable(tf.random_uniform([X_dim, rank], minval=_min_val, maxval=_max_val, seed=_seeds[1]))
        tf_fm_b = tf.Variable(tf.zeros([1]))

        tf_fm_wv = tf.concat(1, [tf_fm_w, tf_fm_v])
        tf_vf_w = [tf.slice(tf_fm_wv, [_i, 0], [1, rank + 1]) for _i in range(13)]
        tf_cf_w = [tf.slice(tf_fm_wv, [offsets[_i], 0], [cat_sizes[_i], rank + 1]) for _i in range(X_feas - 13)]

        tf_vf_x = tf.placeholder(tf.float32, shape=[batch_size, 13])

        tf_vf_mbd = tf.concat(1, [tf.matmul(tf.reshape(tf_vf_x[:, _i], [batch_size, 1]), tf_vf_w[_i]) for _i in range(13)])
        inds = []
        for _i in range(batch_size):
            inds.append([_i, 0])

        # tmp4 = tf_cf_w[0]
        id_vals = tf.placeholder(tf.int64, shape=[batch_size, 26])
        weight_vals = tf.placeholder(tf.float32, shape=[batch_size, 26])
        tmp6 = [tf.SparseTensor(inds, id_vals[:, _i], shape=[batch_size, 1]) for _i in range(X_feas - 13)]
        tmp7 = [tf.SparseTensor(inds, weight_vals[:, _i], shape=[batch_size, 1]) for _i in range(X_feas - 13)]
        tf_cf_mbd = tf.concat(1, [tf.nn.embedding_lookup_sparse(tf_cf_w[_i], tmp6[_i],
                                                                tmp7[_i], combiner='sum') for _i in range(X_feas - 13)])

        tf_fm_b_mbd = tf.reshape(tf.concat(0, [tf.identity(tf_fm_b) for _i in range(batch_size)]), [batch_size, 1])
        tf_mbd = tf.concat(1, [tf_vf_mbd, tf_cf_mbd, tf_fm_b_mbd])

    with tf.Session(graph=graph) as sess:
        feed_dict = {tf_vf_x: [[1.0] * 13] * batch_size}
        feed_dict[id_vals] = np.zeros((batch_size, 26), dtype=np.int32)
        feed_dict[weight_vals] = np.ones((batch_size, 26), dtype=np.float32)

        feed_dict[id_vals][0] = 0
        feed_dict[id_vals][1] = 1
        feed_dict[id_vals][2] = 2

        feed_dict[weight_vals][3] = -1
        feed_dict[weight_vals][4] = 2
        feed_dict[weight_vals][5] = 0

        # print feed_dict
        fetch_list = [tf_mbd]
        l = sess.run(fetch_list, feed_dict=feed_dict)

        for ll in l:
            print np.array(ll)
            print np.array(ll).shape

if __name__ == '__main__':
    if algo == 'LR':
        lr()
    elif algo == 'FM':
        fm()
    elif algo == 'FNN':
        fnn()
