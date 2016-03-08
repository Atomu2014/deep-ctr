import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

max_vals = [65535, 8000, 2330, 746810, 8000, 57199, 5277, 225635, 3565, 14, 310, 25304793, 21836]
cat_sizes = np.array(
    [18576837, 29427, 15127, 7295, 19901, 3, 6465, 1310, 61, 11700067, 622921, 219556, 10, 2209, 9779, 71, 4, 963, 14,
     22022124, 4384510, 15960286, 290588, 10829, 95, 34])
mask = np.where(cat_sizes < 10000)[0]
offsets = [13 + sum(cat_sizes[mask[:i]]) for i in range(len(mask))]
X_dim = 13 + np.sum(cat_sizes[mask])

print len(mask), np.sum(cat_sizes[mask])

fin = None
file_list = ['../data/day_0_scale', '../data/day_0_scale']
file_index = 0
line_index = 0
batch_size = 10000
epoch = 10
sp_indices = []
_learning_rate = 0.5
_lambda = 0.0001
_alpha = 1
_keep_prob = 0.5

for i in range(batch_size):
    for j in range(13 + len(mask)):
        sp_indices.append([i, j])


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
    rows = []
    cols = []
    values = []
    for i in range(start_index, start_index + size):
        line = next(fin)
        if len(line.strip()):
            y, f, x = get_fxy(line)
            rows.extend([row_start + len(labels) for i in range(len(f))])
            cols.extend(f)
            values.extend(x)
            labels.append(y)
        else:
            break

    return labels, rows, cols, values


def get_batch_xy():
    global file_index, line_index, batch_size
    labels, rows, cols, values = get_batch_sparse_tensor(file_list[file_index], line_index, batch_size)
    if len(labels) == batch_size:
        line_index += batch_size
        return np.array(labels), np.array(rows), np.array(cols), np.array(values)

    file_index = (file_index + 1) % len(file_list)
    line_index = batch_size - len(labels)
    l, r, c, v = get_batch_sparse_tensor(file_list[file_index], 0, batch_size - len(labels))
    labels.extend(l)
    rows.extend(r)
    cols.extend(c)
    values.extend(v)
    return np.array(labels), np.array(rows), np.array(cols), np.array(values)

print 'batch_size: %d, learning_rate: %f, alpha: %f, lambda: %f, keep_prob: %f' % (
    batch_size, _learning_rate, _alpha, _lambda, _keep_prob)

with open('../data/day_0_test_x30_concat', 'r') as valid_fin:
    valid_labels = []
    valid_cols = []
    valid_vals = []
    valid_num_row = 0
    for line in valid_fin:
        fields = line.replace(':', ' ').split()
        valid_labels.append(int(fields[0]))
        valid_num_row += 1
        valid_cols.extend([int(fields[i]) for i in range(1, len(fields), 2)])
        valid_vals.extend([float(fields[i]) for i in range(2, len(fields), 2)])
        if valid_num_row == 10000:
            break

valid_rows = []
for i in range(valid_num_row):
    for j in range(13 + len(mask)):
        valid_rows.append([i, j])

graph = tf.Graph()
with graph.as_default():
    tf_sp_id_vals = tf.placeholder(tf.int64, shape=[batch_size * (13 + len(mask))])
    tf_sp_weight_vals = tf.placeholder(tf.float32, shape=[batch_size * (13 + len(mask))])
    tf_sp_ids = tf.SparseTensor(sp_indices, tf_sp_id_vals, shape=[batch_size, 13 + len(mask)])
    tf_sp_weights = tf.SparseTensor(sp_indices, tf_sp_weight_vals, shape=[batch_size, 13 + len(mask)])
    tf_train_labels = tf.placeholder(tf.float32, shape=[batch_size])
    tf_valid_ids = tf.SparseTensor(valid_rows, valid_cols, shape=[len(valid_labels), 13 + len(mask)])
    tf_valid_weights = tf.SparseTensor(valid_rows, valid_vals, shape=[len(valid_labels), 13 + len(mask)])

    weights = tf.Variable(tf.truncated_normal([X_dim, 1]))
    bias = tf.Variable(tf.zeros([1]))

    logits = tf.nn.embedding_lookup_sparse(weights, tf_sp_ids, tf_sp_weights, combiner='sum') + bias
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, tf_train_labels)) + _lambda * tf.nn.l2_loss(
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
        batch_labels, _, batch_cols, batch_values = get_batch_xy()

        feed_dict = {tf_sp_id_vals: batch_cols, tf_sp_weight_vals: batch_values,
                     tf_train_labels: batch_labels}
        _, l, pred = session.run([optimizer, loss, train_pred], feed_dict=feed_dict)
        if step % epoch == 0:
            print 'loss as step %d: %f' % (step, l)
            try:
                batch_auc = roc_auc_score(batch_labels, pred)
                valid_auc = roc_auc_score(valid_labels, valid_pred.eval())
                print 'train-auc: %.4f\teval-auc: %.4f' % (
                    batch_auc, valid_auc)
            except ValueError as e:
                print 'train-auc: None'
