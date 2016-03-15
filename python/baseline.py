import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

max_vals = [65535, 8000, 705, 249000, 7974, 14159, 946, 43970, 5448, 11, 471, 35070017, 8376]
cat_sizes = np.array(
    [631839, 25297, 15534, 7109, 19273, 4, 6754, 1303, 55, 527815, 147691, 110674, 11, 2202, 9023, 63, 5, 946, 15,
     639542, 357567, 600328, 118442, 10217, 78, 34])
mask = np.where(cat_sizes < 100000)[0]
offsets = [13 + sum(cat_sizes[mask[:i]]) for i in range(len(mask))]
X_dim = 13 + np.sum(cat_sizes[mask])
X_feas = 13 + len(mask)

print len(mask), np.sum(cat_sizes[mask])

fin = None
file_list = ['../data/nds.10.shuf.ind', '../data/nds.10.shuf.ind']
file_index = 0
line_index = 0

batch_size = 1
buffer_size = 10000
eval_size = 10000
epoch = 1000

_learning_rate = 0.001
# _alpha = 1
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


print 'batch: %d, epoch: %d, lr: %.4f, lambda: %.4f, stddev: %.4f, keep_prob: %.4f' % (
    batch_size, epoch, _learning_rate, _lambda, _stddev, _keep_prob)

valid_labels = []
valid_cols = []
valid_vals = []
sp_train_inds = []
sp_valid_inds = []
valid_num_row = 0

with open('../data/test.nds.10.shuf.ind', 'r') as valid_fin:
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

valid_sq_vals = np.array(valid_vals) ** 2

for i in range(batch_size):
    for j in range(X_feas):
        sp_train_inds.append([i, j])


def lr_sgd():
    assert (batch_size == 1), 'batch size should be zero'

    graph = tf.Graph()
    with graph.as_default():
        tf_sp_id_vals = tf.placeholder(tf.int64, shape=[X_feas])
        tf_sp_weight_vals = tf.placeholder(tf.float32, shape=[X_feas])
        tf_sp_ids = tf.SparseTensor(sp_train_inds, tf_sp_id_vals, shape=[1, X_feas])
        tf_sp_weights = tf.SparseTensor(sp_train_inds, tf_sp_weight_vals, shape=[1, X_feas])
        tf_train_label = tf.placeholder(tf.float32)
        tf_sp_valid_ids = tf.SparseTensor(sp_valid_inds, valid_cols, shape=[1, X_feas])
        tf_sp_valid_weights = tf.SparseTensor(sp_valid_inds, valid_vals, shape=[1, X_feas])

        weights = tf.Variable(tf.truncated_normal([X_dim, 1], stddev=_stddev))
        bias = tf.Variable(0.0)

        logits = tf.nn.embedding_lookup_sparse(weights, tf_sp_ids, tf_sp_weights, combiner='sum') + bias
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, tf_train_label) + _lambda * (
            tf.nn.l2_loss(weights) + tf.nn.l2_loss(bias))
        optimizer = tf.train.GradientDescentOptimizer(_learning_rate).minimize(loss)

        train_pred = tf.sigmoid(logits)
        valid_pred = tf.sigmoid(
            tf.nn.embedding_lookup_sparse(weights, tf_sp_valid_ids, tf_sp_valid_weights, combiner='sum') + bias)

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print 'initialized'
        step = 0
        while True:
            label, _, cols, values = get_batch_xy(buffer_size)
            for i in range(label.shape[0]):
                step += 1
                feed_dict = {tf_sp_id_vals: cols[i * X_feas: (i + 1) * X_feas],
                             tf_sp_weight_vals: values[i * X_feas: (i + 1) * X_feas], tf_train_label: label[i]}
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

    def factorization(sp_ids, sp_weights, sp_weights2):
        yhat = tf.nn.embedding_lookup_sparse(W, sp_ids, sp_weights, combiner='sum') + b
        Vx = tf.nn.embedding_lookup_sparse(V, sp_ids, sp_weights, combiner='sum')
        V2x2 = tf.nn.embedding_lookup_sparse(tf.square(V), sp_ids, sp_weights2, combiner='sum')
        yhat += 0.5 * tf.reshape(tf.reduce_sum(tf.matmul(Vx, Vx, transpose_b=True), 1) - tf.reduce_sum(V2x2, 1),
                                 shape=[-1, 1])
        return yhat

    with graph.as_default():
        tf_sp_id_vals = tf.placeholder(tf.int64, shape=[X_feas])
        tf_sp_weight_vals = tf.placeholder(tf.float32, shape=[X_feas])
        tf_sp_ids = tf.SparseTensor(sp_train_inds, tf_sp_id_vals, shape=[1, X_feas])
        tf_sp_weights = tf.SparseTensor(sp_train_inds, tf_sp_weight_vals, shape=[1, X_feas])
        tf_train_label = tf.placeholder(tf.float32)
        tf_sp_weight2_vals = tf.placeholder(tf.float32, shape=[X_feas])
        tf_sp_weights2 = tf.SparseTensor(sp_train_inds, tf_sp_weight2_vals, shape=[1, X_feas])
        tf_sp_valid_ids = tf.SparseTensor(sp_valid_inds, valid_cols, shape=[valid_num_row, X_feas])
        tf_sp_valid_weights = tf.SparseTensor(sp_valid_inds, valid_vals, shape=[valid_num_row, X_feas])
        tf_sp_valid_weights2 = tf.SparseTensor(sp_valid_inds, valid_sq_vals, shape=[valid_num_row, X_feas])

        W = tf.Variable(tf.truncated_normal([X_dim, 1], stddev=_stddev))
        b = tf.Variable(0.0)
        V = tf.Variable(tf.truncated_normal([X_dim, _rank], stddev=_stddev))

        logit = factorization(tf_sp_ids, tf_sp_weights, tf_sp_weights2)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logit, tf_train_label) + _lambda * (
            tf.nn.l2_loss(W) + tf.nn.l2_loss(V) + tf.nn.l2_loss(b))
        optimizer = tf.train.GradientDescentOptimizer(_learning_rate).minimize(loss)
        train_pred = tf.sigmoid(logit)
        valid_logits = factorization(tf_sp_valid_ids, tf_sp_valid_weights, tf_sp_valid_weights2)
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

                feed_dict = {tf_sp_id_vals: _cols, tf_sp_weight_vals: _vals, tf_sp_weight2_vals: _sqvals,
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
