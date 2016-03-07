import linecache

import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score

from dl_utils import load_svmlight

max_vals = [65535, 8000, 2330, 746810, 8000, 57199, 5277, 225635, 3565, 14, 310, 25304793, 21836]
cat_sizes = np.array(
    [18576837, 29427, 15127, 7295, 19901, 3, 6465, 1310, 61, 11700067, 622921, 219556, 10, 2209, 9779, 71, 4, 963, 14,
     22022124, 4384510, 15960286, 290588, 10829, 95, 34])
mask = np.where(cat_sizes < 10000)[0]
offsets = [13 + sum(cat_sizes[mask[:i]]) for i in range(len(mask))]
X_dim = 13 + np.sum(cat_sizes[mask])

valid_dataset, valid_labels = load_svmlight('../data/day_0_test_x30_concat')

file_list = ['../data/day_0_train_x30', '../data/day_0_train_x30']
file_index = 0
line_index = 0


def get_fxy(line):
    fields = line.split('\t')
    cats = fields[14:]
    feats = range(13)
    feats.extend([int(cats[mask[i]]) + offsets[i] for i in range(len(mask))])
    x = [float(fields[i]) / max_vals[i - 1] for i in range(1, 14)]
    x.extend([1] * len(mask))
    return fields[0], feats, x


def get_batch(batch_size):
    global line_index, file_index
    labels = []
    rows = []
    cols = []
    vals = []
    num_rows = 0
    for i in range(line_index, line_index + batch_size):
        line = linecache.getline(file_list[file_index], i)
        num_rows += 1
        if line.strip() != '':
            y, f, x = get_fxy(line)
            labels.append(y)
            rows.extend([num_rows - 1] * len(f))
            cols.extend(f)
            vals.extend(x)
        else:
            break

    if num_rows == batch_size:
        line_index += batch_size
        return labels, csr_matrix((vals, (rows, cols)))

    file_index += 1
    file_index %= len(file_list)
    linecache.clearcache()
    line_index = batch_size - len(labels)

    for i in range(0, batch_size - len(labels)):
        line = linecache.getline(file_list[file_index], i)
        num_rows += 1
        if line.strip() != '':
            y, f, x = get_fxy(line)
            labels.append(y)
            rows.extend([num_rows - 1] * len(f))
            cols.extend(f)
            vals.extend(x)
        else:
            break

    return labels, csr_matrix((vals, (rows, cols)))


def auc(predictions, labels):
    return roc_auc_score(labels, predictions)


def lr():
    _learning_rate = 0.001
    _l2_param = 0.001
    _keep_prob = 0.5
    _batch_size = 100

    print 'batch_size: %d, learning_rate: %f, l2_param: %f, keep_prob: %f' % (
        _batch_size, _learning_rate, _l2_param, _keep_prob)

    graph = tf.Graph()
    with graph.as_default():
        tf_train_dataset = tf.placeholder(tf.float32, shape=(_batch_size, X_dim))
        tf_train_labels = tf.placeholder(tf.float32, shape=(_batch_size, 1))
        tf_valid_dataset = tf.constant(valid_dataset)
        # tf_test_dataset = tf.constant(test_dataset)

        weights = tf.Variable(tf.truncated_normal([X_dim, 1]))
        bias = tf.Variable(tf.zeros([1]))

        logits = tf.matmul(tf_train_dataset, weights, a_is_sparse=True) + bias
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, tf_train_labels))

        optimizer = tf.train.GradientDecentOptimizer(_learning_rate).minimize(loss)

        train_pred = tf.sigmoid(logits)
        valid_pred = tf.sigmoid(tf.matmul(tf_valid_dataset, weights, a_is_sparse=True) + bias)
        # test_pred = tf.sigmoid(tf.matmul(tf_test_dataset, weights, a_is_sparse=True) + bias)

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print 'initialized'

        step = 0
        while True:
            step += 1
            batch_labels, batch_data = get_batch(_batch_size)

            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, pred = session.run([optimizer, loss, train_pred], feed_dict=feed_dict)
            if step % 100 == 0:
                print 'loss as step %d: %f' % (step, l)
                print 'train-auc: %f%%\teval-auc: %f%%' % (
                    roc_auc_score(batch_labels, pred), roc_auc_score(valid_labels, valid_pred))
