import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

max_vals = [65535, 8000, 2330, 746810, 8000, 57199, 5277, 225635, 3565, 14, 310, 25304793, 21836]
cat_sizes = np.array(
    [18576837, 29427, 15127, 7295, 19901, 3, 6465, 1310, 61, 11700067, 622921, 219556, 10, 2209, 9779, 71, 4, 963, 14,
     22022124, 4384510, 15960286, 290588, 10829, 95, 34])
mask = np.where(cat_sizes < 1000)[0]
offsets = [13 + sum(cat_sizes[mask[:i]]) for i in range(len(mask))]
X_dim = 13 + np.sum(cat_sizes[mask])

print len(mask), np.sum(cat_sizes[mask])

fin = None
file_list = ['../data/day_0_scale', '../data/day_0_scale']
file_index = 0
line_index = 0

batch_size = 1
epoch = 1000

_learning_rate = 0.001
_alpha = 1
_lambda = 0.001
_keep_prob = 0.5
_stddev = 0.01
_rank = 20


def get_fxy(line):
    fields = line.split('\t')
    y = int(fields[0])
    cats = fields[14:]
    f = range(13)
    f.extend([int(cats[mask[i]]) + offsets[i] for i in range(len(mask))])
    x = [float(fields[i]) / max_vals[i - 1] for i in range(1, 14)]
    x.extend([1] * len(mask))
    return y, f, x


def get_batch_sparse_tensor(file_name, start_index, size, row_start=0):
    global fin
    if fin is None:
        fin = open(file_name, 'r')
    labels = []
    cols = []
    values = []
    indices = []
    row_num = row_start
    for i in range(start_index, start_index + size):
        try:
            line = next(fin)
            if len(line.strip()):
                y, f, x = get_fxy(line)
                indices.extend([[row_num, c] for c in f])
                cols.extend(f)
                values.extend(x)
                labels.append(y)
                row_num += 1
            else:
                break
        except StopIteration as e:
            print e
            fin = None
            break

    return labels, indices, cols, values


def get_batch_xy(size):
    global file_index, line_index
    labels, indices, cols, values = get_batch_sparse_tensor(file_list[file_index], line_index, size)
    if len(labels) == size:
        line_index += size
        return np.array(labels), np.array(indices), np.array(cols), np.array(values)

    file_index = (file_index + 1) % len(file_list)
    line_index = size - len(labels)
    l, i, c, v = get_batch_sparse_tensor(file_list[file_index], 0, size - len(labels))
    labels.extend(l)
    indices.extend(i)
    cols.extend(c)
    values.extend(v)
    return np.array(labels), np.array(indices), np.array(cols), np.array(values)


print 'batch: %d, epoch: %d, lr: %.4f, lambda: %.4f, stddev: %.4f, keep_prob: %.4f, alpha: %.4f' % (
    batch_size, epoch, _learning_rate, _lambda, _stddev, _keep_prob, _alpha)

with open('../data/day_0_test_x30', 'r') as valid_fin:
    valid_labels = []
    valid_inds = []
    valid_cols = []
    valid_vals = []
    valid_num_row = 0
    for line in valid_fin:
        y, f, x = get_fxy(line)
        valid_inds.extend([[valid_num_row, c] for c in f])
        valid_cols.extend(f)
        valid_vals.extend(x)
        valid_labels.append(y)
        valid_num_row += 1
        if valid_num_row == 10000:
            break

valid_rows = []
for i in range(valid_num_row):
    for j in range(13 + len(mask)):
        valid_rows.append([i, j])

sp_indices = []
for i in range(batch_size):
    for j in range(13 + len(mask)):
        sp_indices.append([i, j])


def lr_bgd():
    graph = tf.Graph()
    with graph.as_default():
        tf_sp_id_vals = tf.placeholder(tf.int64, shape=[batch_size * (13 + len(mask))])
        tf_sp_weight_vals = tf.placeholder(tf.float32, shape=[batch_size * (13 + len(mask))])
        tf_sp_ids = tf.SparseTensor(sp_indices, tf_sp_id_vals, shape=[batch_size, 13 + len(mask)])
        tf_sp_weights = tf.SparseTensor(sp_indices, tf_sp_weight_vals, shape=[batch_size, 13 + len(mask)])
        tf_train_labels = tf.placeholder(tf.float32, shape=[batch_size])
        tf_valid_ids = tf.SparseTensor(valid_rows, valid_cols, shape=[valid_num_row, 13 + len(mask)])
        tf_valid_weights = tf.SparseTensor(valid_rows, valid_vals, shape=[valid_num_row, 13 + len(mask)])

        weights = tf.Variable(tf.truncated_normal([X_dim, 1], stddev=_stddev))
        bias = tf.Variable(0.0)

        logits = tf.nn.embedding_lookup_sparse(weights, tf_sp_ids, tf_sp_weights, combiner='sum') + bias
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits, tf_train_labels)) + _lambda * tf.nn.l2_loss(
            weights)

        optimizer = tf.train.GradientDescentOptimizer(_learning_rate).minimize(loss)

        train_pred = tf.sigmoid(logits)
        valid_pred = tf.sigmoid(
            tf.nn.embedding_lookup_sparse(weights, tf_valid_ids, tf_valid_weights, combiner='sum') + bias)

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print 'initialized'

        step = 0
        while True:
            step += 1
            batch_labels, batch_cols, batch_values = get_batch_xy(batch_size)

            feed_dict = {tf_sp_id_vals: batch_cols, tf_sp_weight_vals: batch_values,
                         tf_train_labels: batch_labels}
            _, l, pred = session.run([optimizer, loss, train_pred], feed_dict=feed_dict)
            if step % epoch == 0:
                print 'loss as step %d: %f' % (step, l)
                try:
                    # batch_auc = roc_auc_score(batch_labels, pred)
                    valid_auc = roc_auc_score(valid_labels, valid_pred.eval())
                    print 'eval-auc: %.4f' % valid_auc
                except ValueError as e:
                    print 'None'


def lr_sgd():
    assert (batch_size == 1), 'batch size should be zero'

    graph = tf.Graph()
    with graph.as_default():
        tf_sp_id_vals = tf.placeholder(tf.int64, shape=[13 + len(mask)])
        tf_sp_weight_vals = tf.placeholder(tf.float32, shape=[13 + len(mask)])
        tf_sp_ids = tf.SparseTensor(sp_indices, tf_sp_id_vals, shape=[batch_size, 13 + len(mask)])
        tf_sp_weights = tf.SparseTensor(sp_indices, tf_sp_weight_vals, shape=[batch_size, 13 + len(mask)])
        tf_train_label = tf.placeholder(tf.float32, shape=[batch_size])
        tf_valid_ids = tf.SparseTensor(valid_rows, valid_cols, shape=[batch_size, 13 + len(mask)])
        tf_valid_weights = tf.SparseTensor(valid_rows, valid_vals, shape=[batch_size, 13 + len(mask)])

        weights = tf.Variable(tf.truncated_normal([X_dim, 1], stddev=_stddev))
        bias = tf.Variable(0.0)

        logits = tf.nn.embedding_lookup_sparse(weights, tf_sp_ids, tf_sp_weights, combiner='sum') + bias
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, tf_train_label) + _lambda * (
            tf.nn.l2_loss(weights) + tf.square(bias))

        optimizer = tf.train.GradientDescentOptimizer(_learning_rate).minimize(loss)

        train_pred = tf.sigmoid(logits)
        valid_pred = tf.sigmoid(
            tf.nn.embedding_lookup_sparse(weights, tf_valid_ids, tf_valid_weights, combiner='sum') + bias)

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print 'initialized'

        step = 0
        while True:
            step += 1
            _label, _cols, _values = get_batch_xy(batch_size)

            feed_dict = {tf_sp_id_vals: _cols, tf_sp_weight_vals: _values, tf_train_label: _label}
            _, l, pred = session.run([optimizer, loss, train_pred], feed_dict=feed_dict)
            if step % epoch == 0:
                print 'loss as step %d: %f' % (step, l)
                try:
                    # batch_auc = roc_auc_score(_label, pred)
                    valid_auc = roc_auc_score(valid_labels, valid_pred.eval())
                    print 'eval-auc: %.4f' % valid_auc
                except ValueError as e:
                    print 'None'


def fm_sgd():
    assert (batch_size == 1), 'batch size should be zero'
    print 'rank: %d' % _rank

    graph = tf.Graph()

    def factorization(sp_ids, sp_weights, x):
        yhat = tf.nn.embedding_lookup_sparse(W, sp_ids, sp_weights, combiner='sum') + b
        # wb_shape = tf.shape(yhat)
        vx = tf.nn.embedding_lookup_sparse(V, sp_ids, sp_weights, combiner='sum')
        # vx = tf.matmul(x, V)
        # vx_shape = tf.shape(vx)
        # vxvx = tf.matmul(vx, vx, transpose_b=True)
        # vxvx_shape = tf.shape(vxvx)
        # vxvxs = tf.reduce_sum(vxvx, 1)
        # vxvxs_shape = tf.shape(vxvxs)
        # x2 = tf.square(x)
        # x2_shape = tf.shape(x2)
        # V2 = tf.square(V)
        # V2_shape = tf.shape(V2)
        # x2v2 = tf.matmul(x2, V2)
        # x2v2_shape = tf.shape(x2v2)
        # x2v2s = tf.reduce_sum(x2v2, 1)
        # x2v2s_shape = tf.shape(x2v2s)
        yhat += 0.5 * tf.reshape(
            tf.reduce_sum(tf.matmul(vx, vx, transpose_b=True), 1) - tf.reduce_sum(tf.matmul(tf.square(x), tf.square(V)),
                                                                                  1), shape=[-1, 1])
        return yhat
        # , wb_shape, vx_shape, vxvx_shape, vxvxs_shape, x2_shape, V2_shape, x2v2_shape, x2v2s_shape

    with graph.as_default():
        tf_sp_id_vals = tf.placeholder(tf.int64, shape=[13 + len(mask)])
        tf_sp_weight_vals = tf.placeholder(tf.float32, shape=[13 + len(mask)])
        tf_sp_ids = tf.SparseTensor(sp_indices, tf_sp_id_vals, shape=[batch_size, 13 + len(mask)])
        tf_sp_weights = tf.SparseTensor(sp_indices, tf_sp_weight_vals, shape=[batch_size, 13 + len(mask)])
        tf_train_label = tf.placeholder(tf.float32, shape=[1])
        tf_train_indices = tf.placeholder(tf.int64, shape=[batch_size * (13 + len(mask)), 2])
        tf_train = tf.sparse_to_dense(tf_train_indices, [batch_size, X_dim], tf_sp_weight_vals)
        tf_valid_ids = tf.SparseTensor(valid_rows, valid_cols, shape=[valid_num_row, 13 + len(mask)])
        tf_valid_weights = tf.SparseTensor(valid_rows, valid_vals, shape=[valid_num_row, 13 + len(mask)])
        tf_valid = tf.sparse_to_dense(valid_inds, [valid_num_row, X_dim], valid_vals)

        W = tf.Variable(tf.truncated_normal([X_dim, 1], stddev=_stddev))
        b = tf.Variable(0.0)
        V = tf.Variable(tf.truncated_normal([X_dim, _rank], stddev=_stddev))

        logit = factorization(tf_sp_ids, tf_sp_weights, tf_train)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logit, tf_train_label) + _lambda * (
            tf.nn.l2_loss(W) + tf.nn.l2_loss(V) + tf.nn.l2_loss(b))
        optimizer = tf.train.GradientDescentOptimizer(_learning_rate).minimize(loss)
        train_pred = tf.sigmoid(logit)
        valid_logits = factorization(tf_valid_ids, tf_valid_weights, tf_valid)
        valid_pred = tf.sigmoid(valid_logits)
        # vp_shape = tf.shape(valid_pred)

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print 'initialized'

        step = 0
        while True:
            step += 1
            _label, _ind, _cols, _values = get_batch_xy(1)

            feed_dict = {tf_sp_id_vals: _cols, tf_sp_weight_vals: _values, tf_train_label: _label,
                         tf_train_indices: _ind}
            _, l, pred, lg = session.run([optimizer, loss, train_pred, logit], feed_dict=feed_dict)
            if step % epoch == 0:
                print 'loss as step %d: %f' % (step, l)
                try:
                    # print 'wb', wb_shape.eval()
                    # print 'vx', vx_shape.eval()
                    # print 'vxvx', vxvx_shape.eval()
                    # print 'vxvxs', vxvxs_shape.eval()
                    # print 'x2', x2_shape.eval()
                    # print 'V2', V2_shape.eval()
                    # print 'x2v2', x2v2_shape.eval()
                    # print 'x2v2s', x2v2s_shape.eval()
                    # print 'vp', vp_shape.eval()
                    valid_auc = roc_auc_score(valid_labels, valid_pred.eval())
                    print 'eval-auc: %.4f' % valid_auc
                except ValueError as e:
                    print 'None'


if __name__ == '__main__':
    fm_sgd()
    # lr_sgd()
