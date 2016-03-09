import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

max_vals = [65535, 8000, 2330, 746810, 8000, 57199, 5277, 225635, 3565, 14, 310, 25304793, 21836]
cat_sizes = np.array(
    [18576837, 29427, 15127, 7295, 19901, 3, 6465, 1310, 61, 11700067, 622921, 219556, 10, 2209, 9779, 71, 4, 963, 14,
     22022124, 4384510, 15960286, 290588, 10829, 95, 34])
mask = np.where(cat_sizes < 10000000)[0]
offsets = [13 + sum(cat_sizes[mask[:i]]) for i in range(len(mask))]
X_dim = 13 + np.sum(cat_sizes[mask])

print len(mask), np.sum(cat_sizes[mask])

fin = None
file_list = ['../data/day_0_scale', '../data/day_0_scale']
file_index = 0
line_index = 0

batch_size = 1
epoch = 1000

_learning_rate = 0.01
_alpha = 1
_lambda = 0
_keep_prob = 0.5
_stddev = 0.1
_rank = 10


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
    for i in range(start_index, start_index + size):
        try:
            line = next(fin)
            if len(line.strip()):
                y, f, x = get_fxy(line)
                cols.extend(f)
                values.extend(x)
                labels.append(y)
            else:
                break
        except StopIteration as e:
            print e
            fin = None
            break

    return labels, cols, values


def get_batch_xy(size):
    global file_index, line_index
    labels, cols, values = get_batch_sparse_tensor(file_list[file_index], line_index, size)
    if len(labels) == size:
        line_index += size
        return np.array(labels), np.array(cols), np.array(values)

    file_index = (file_index + 1) % len(file_list)
    line_index = size - len(labels)
    l, c, v = get_batch_sparse_tensor(file_list[file_index], 0, size - len(labels))
    labels.extend(l)
    cols.extend(c)
    values.extend(v)
    return np.array(labels), np.array(cols), np.array(values)


print 'batch_size: %d, epoch: %d, learning_rate: %f, alpha: %f, lambda: %f, keep_prob: %f, stddev: %f' % (
    batch_size, epoch, _learning_rate, _alpha, _lambda, _keep_prob, _stddev)

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
                    print 'train-auc: None'


def fm_sgd():
    assert (batch_size == 1), 'batch size should be zero'
    u = np.ones((X_dim,))

    graph = tf.Graph()

    # def factorization(sp_ids, sp_weights):
    #     U = tf.constant(u)
    #
    #     yhat = tf.nn.embedding_lookup_sparse(W, tf_sp_ids, tf_sp_weights, combiner='sum') + b
    #     for i in range(tf.shape(V).eval()[1]):
    #         vf = tf.slice(V, [0, i], [tf.shape(V).eval()[0], 1])
    #         vv = tf.matmul(vf, vf)
    #         ux = tf.nn.embedding_lookup_sparse(U, sp_ids, sp_weights, combiner='sum')
    #         xx = tf.nn.embedding_lookup_sparse(ux, sp_ids, sp_weights, combiner='sum')
    #         yhat += 0.5 * (tf.square(tf.nn.embedding_lookup_sparse(vf, sp_ids, sp_weights, combiner='sum')) - tf.matmul(vv, xx))
    #
    #     return yhat

    with graph.as_default():
        tf_sp_id_vals = tf.placeholder(tf.int64, shape=[13 + len(mask)])
        tf_sp_weight_vals = tf.placeholder(tf.float32, shape=[13 + len(mask)])
        tf_sp_ids = tf.SparseTensor(sp_indices, tf_sp_id_vals, shape=[batch_size, 13 + len(mask)])
        tf_sp_weights = tf.SparseTensor(sp_indices, tf_sp_weight_vals, shape=[batch_size, 13 + len(mask)])
        tf_train_label = tf.placeholder(tf.float32, shape=[1])
        # tf_valid_ids = tf.SparseTensor(valid_rows, valid_cols, shape=[batch_size, 13 + len(mask)])
        # tf_valid_weights = tf.SparseTensor(valid_rows, valid_vals, shape=[batch_size, 13 + len(mask)])

        W = tf.Variable(tf.truncated_normal([X_dim, 1], stddev=_stddev))
        b = tf.Variable(0.0)
        V = tf.Variable(tf.truncated_normal([X_dim, _rank], stddev=_stddev))
        U = tf.Variable(tf.ones([X_dim]))

        wxb = tf.nn.embedding_lookup_sparse(W, tf_sp_ids, tf_sp_weights, combiner='sum') + b
        ux = tf.nn.embedding_lookup_sparse(U, tf_sp_ids, tf_sp_weights, combiner='sum')
        ux4d = tf.reshape(ux, [1, X_dim, 1, 1])
        v4d = tf.reshape(V, [1, X_dim, _rank, 1])
        conv_x = tf.nn.conv2d(ux4d, ux4d, [1, 1, 1, 1], 'VALID')
        conv_v = tf.nn.conv2d(v4d, v4d, [1, 1, 1, 1], 'VALID')
        vx = tf.nn.embedding_lookup_sparse(V, tf_sp_ids, tf_sp_weights, combiner='sum')
        vx4d = tf.reshape(vx, [1, _rank, 1, 1])
        conv_vx = tf.nn.conv2d(vx4d, vx4d, [1, 1, 1, 1], 'VALID')
        logits = tf.sigmoid(wxb + 0.5 * (conv_vx - conv_x * conv_v))
        # logits = tf.sigmoid(factorization(tf_sp_ids, tf_sp_weights))

        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, tf_train_label) + _lambda * (
            tf.nn.l2_loss(W) + tf.nn.l2_loss(V) + tf.square(b))

        optimizer = tf.train.GradientDescentOptimizer(_learning_rate).minimize(loss)
        # train_pred = tf.sigmoid(logits)
        # valid_pred = tf.sigmoid(
        #     tf.nn.embedding_lookup_sparse(weights, tf_valid_ids, tf_valid_weights, combiner='sum') + bias)

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print 'initialized'

        step = 0
        while True:
            step += 1
            _label, _cols, _values = get_batch_xy(1)

            feed_dict = {tf_sp_id_vals: _cols, tf_sp_weight_vals: _values, tf_train_label: _label}
            # _, l, pred = session.run([optimizer, loss, train_pred], feed_dict=feed_dict)
            # if step % epoch == 0:
            #     print 'loss as step %d: %f' % (step, l)
            #     try:
            # valid_auc = roc_auc_score(valid_labels, valid_pred.eval())
            # print 'eval-auc: %.4f' % valid_auc
            # except ValueError as e:
            #     print 'train-auc: None'


if __name__ == '__main__':
    fm_sgd()
    # lr_sgd()
