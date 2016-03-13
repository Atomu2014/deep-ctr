import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

max_vals = [65535, 8000, 2330, 746810, 8000, 57199, 5277, 225635, 3565, 14, 310, 25304793, 21836]
cat_sizes = np.array(
    [18576837, 29427, 15127, 7295, 19901, 3, 6465, 1310, 61, 11700067, 622921, 219556, 10, 2209, 9779, 71, 4, 963, 14,
     22022124, 4384510, 15960286, 290588, 10829, 95, 34])
mask = np.where(cat_sizes < 100000)[0]
offsets = [13 + sum(cat_sizes[mask[:i]]) for i in range(len(mask))]
X_dim = 13 + np.sum(cat_sizes[mask])
X_feas = 13 + len(mask)

print len(mask), np.sum(cat_sizes[mask])

fin = None
file_list = ['../data/day_0_scale', '../data/day_0_scale']
file_index = 0
line_index = 0

batch_size = 1
buffer_size = 10000
epoch = 1000

_learning_rate = 0.001
_alpha = 1
_lambda = 0.001
_keep_prob = 0.5
_stddev = 0.1
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
    rows = []
    row_num = row_start
    for i in range(start_index, start_index + size):
        try:
            line = next(fin)
            if len(line.strip()):
                y, f, x = get_fxy(line)
                rows.extend([row_num] * len(f))
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


print 'batch: %d, epoch: %d, lr: %.4f, lambda: %.4f, stddev: %.4f, keep_prob: %.4f, alpha: %.4f' % (
    batch_size, epoch, _learning_rate, _lambda, _stddev, _keep_prob, _alpha)

with open('../data/day_0_test_x30', 'r') as valid_fin:
    valid_labels = []
    valid_cols = []
    valid_vals = []
    valid_num_row = 0
    for line in valid_fin:
        y, f, x = get_fxy(line)
        valid_cols.extend(f)
        valid_vals.extend(x)
        valid_labels.append(y)
        valid_num_row += 1
        if valid_num_row == 10000:
            break

valid_inds = []
for i in range(valid_num_row):
    for j in range(X_feas):
        valid_inds.append([i, j])

valid_sq_vals = np.array(valid_vals) ** 2

sp_indices = []
for i in range(batch_size):
    for j in range(X_feas):
        sp_indices.append([i, j])


def get_sparse_indices(rows, cols):
    return np.array([[rows[i], cols[i]] for i in range(len(rows))])


def lr_sgd():
    assert (batch_size == 1), 'batch size should be zero'

    graph = tf.Graph()
    with graph.as_default():
        tf_sp_id_vals = tf.placeholder(tf.int64, shape=[X_feas])
        tf_sp_weight_vals = tf.placeholder(tf.float32, shape=[X_feas])
        tf_sp_ids = tf.SparseTensor(sp_indices, tf_sp_id_vals, shape=[1, X_feas])
        tf_sp_weights = tf.SparseTensor(sp_indices, tf_sp_weight_vals, shape=[1, X_feas])
        tf_train_label = tf.placeholder(tf.float32)
        tf_valid_ids = tf.SparseTensor(valid_inds, valid_cols, shape=[1, X_feas])
        tf_valid_weights = tf.SparseTensor(valid_inds, valid_vals, shape=[1, X_feas])

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
            label, _, cols, values = get_batch_xy(buffer_size)
            for i in range(label.shape[0]):
                step += 1
                feed_dict = {tf_sp_id_vals: cols[i * X_feas: (i + 1) * X_feas],
                             tf_sp_weight_vals: values[i * X_feas: (i + 1) * X_feas],
                             tf_train_label: label[i]}
                _, l, pred = session.run([optimizer, loss, train_pred], feed_dict=feed_dict)
                if step % epoch == 0:
                    print 'loss as step %d: %f' % (step, l)
                    try:
                        valid_auc = roc_auc_score(valid_labels, valid_pred.eval())
                        print 'eval-auc: %.4f' % valid_auc
                    except ValueError as e:
                        print 'None'


def fm_sgd():
    assert (batch_size == 1), 'batch size should be zero'
    print 'rank: %d' % _rank

    graph = tf.Graph()

    def factorization(sp_ids, sp_weights, sp_sq_weights):
        yhat = tf.nn.embedding_lookup_sparse(W, sp_ids, sp_weights, combiner='sum') + b
        vx = tf.nn.embedding_lookup_sparse(V, sp_ids, sp_weights, combiner='sum')
        V2 = tf.square(V)
        V2x2 = tf.nn.embedding_lookup_sparse(V2, sp_ids, sp_sq_weights, combiner='sum')
        yhat += 0.5 * tf.reshape(
            tf.reduce_sum(tf.matmul(vx, vx, transpose_b=True), 1) - tf.reduce_sum(
                V2x2,
                1), shape=[-1, 1])
        return yhat

    with graph.as_default():
        tf_sp_id_vals = tf.placeholder(tf.int64, shape=[X_feas])
        tf_sp_weight_vals = tf.placeholder(tf.float32, shape=[X_feas])
        tf_sp_ids = tf.SparseTensor(sp_indices, tf_sp_id_vals, shape=[1, X_feas])
        tf_sp_weights = tf.SparseTensor(sp_indices, tf_sp_weight_vals, shape=[1, X_feas])
        tf_train_label = tf.placeholder(tf.float32)
        tf_sp_sq_weight_vals = tf.placeholder(tf.float32, shape=[X_feas])
        tf_sp_sq_weights = tf.SparseTensor(sp_indices, tf_sp_sq_weight_vals, shape=[1, X_feas])
        tf_valid_ids = tf.SparseTensor(valid_inds, valid_cols, shape=[valid_num_row, X_feas])
        tf_valid_weights = tf.SparseTensor(valid_inds, valid_vals, shape=[valid_num_row, X_feas])
        tf_valid_sq_weights = tf.SparseTensor(valid_inds, valid_sq_vals, shape=[valid_num_row, X_feas])

        W = tf.Variable(tf.truncated_normal([X_dim, 1], stddev=_stddev))
        b = tf.Variable(0.0)
        V = tf.Variable(tf.truncated_normal([X_dim, _rank], stddev=_stddev))

        logit = factorization(tf_sp_ids, tf_sp_weights, tf_sp_sq_weights)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logit, tf_train_label) + _lambda * (
            tf.nn.l2_loss(W) + tf.nn.l2_loss(V) + tf.nn.l2_loss(b))
        optimizer = tf.train.GradientDescentOptimizer(_learning_rate).minimize(loss)
        train_pred = tf.sigmoid(logit)
        valid_logits = factorization(tf_valid_ids, tf_valid_weights, tf_valid_sq_weights)
        valid_pred = tf.sigmoid(valid_logits)

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print 'initialized'
        step = 0
        while True:
            label, rows, cols, values = get_batch_xy(buffer_size)
            for i in range(label.shape[0]):
                step += 1
                _label = label[i]
                _cols = cols[i * X_feas: (i + 1) * X_feas]
                _vals = values[i * X_feas: (i + 1) * X_feas]
                _sqvals = _vals ** 2

                feed_dict = {tf_sp_id_vals: _cols, tf_sp_weight_vals: _vals, tf_sp_sq_weight_vals: _sqvals,
                             tf_train_label: _label}
                _, l, pred = session.run([optimizer, loss, train_pred], feed_dict=feed_dict)
                if step % epoch == 0:
                    print 'loss as step %d: %f' % (step, l)
                    try:
                        valid_auc = roc_auc_score(valid_labels, valid_pred.eval())
                        print 'eval-auc: %.4f' % valid_auc
                    except ValueError as e:
                        print 'None'


if __name__ == '__main__':
    fm_sgd()
    # lr_sgd()
