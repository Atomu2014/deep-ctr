import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score

max_vals = [65535, 8000, 2330, 746810, 8000, 57199, 5277, 225635, 3565, 14, 310, 25304793, 21836]
cat_sizes = np.array(
    [18576837, 29427, 15127, 7295, 19901, 3, 6465, 1310, 61, 11700067, 622921, 219556, 10, 2209, 9779, 71, 4, 963, 14,
     22022124, 4384510, 15960286, 290588, 10829, 95, 34])
mask = np.where(cat_sizes < 10000)[0]
offsets = [13 + sum(cat_sizes[mask[:i]]) for i in range(len(mask))]
X_dim = 13 + np.sum(cat_sizes[mask])

# valid_dataset, valid_labels = load_svmlight('../data/day_0_test_x30_concat')

fin = None
file_list = ['../data/day_0_scale', '../data/day_0_scale']
file_index = 0
line_index = 0
batch_size = 100
sp_indices = []
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


# def get_batch_csr(batch_size):
#     global line_index, file_index
#     labels = []
#     rows = []
#     cols = []
#     vals = []
#     num_rows = 0
#     for i in range(line_index, line_index + batch_size):
#         line = linecache.getline(file_list[file_index], i)
#         num_rows += 1
#         if line.strip() != '':
#             y, f, x = get_fxy(line)
#             labels.append(y)
#             rows.extend([num_rows - 1] * len(f))
#             cols.extend(f)
#             vals.extend(x)
#         else:
#             break
#
#     if num_rows == batch_size:
#         line_index += batch_size
#         return labels, csr_matrix((vals, (rows, cols)))
#
#     file_index += 1
#     file_index %= len(file_list)
#     linecache.clearcache()
#     line_index = batch_size - len(labels)
#
#     for i in range(0, batch_size - len(labels)):
#         line = linecache.getline(file_list[file_index], i)
#         num_rows += 1
#         if line.strip() != '':
#             y, f, x = get_fxy(line)
#             labels.append(y)
#             rows.extend([num_rows - 1] * len(f))
#             cols.extend(f)
#             vals.extend(x)
#         else:
#             break
#
#     return labels, csr_matrix((vals, (rows, cols)))


_learning_rate = 0.001
_l2_param = 0.001
_keep_prob = 0.5

print 'batch_size: %d, learning_rate: %f, l2_param: %f, keep_prob: %f' % (
    batch_size, _learning_rate, _l2_param, _keep_prob)

graph = tf.Graph()
with graph.as_default():
    tf_sp_id_vals = tf.placeholder(tf.int64, shape=[batch_size * (13 + len(mask))])
    tf_sp_weight_vals = tf.placeholder(tf.float32, shape=[batch_size * (13 + len(mask))])
    tf_sp_ids = tf.SparseTensor(sp_indices, tf_sp_id_vals, shape=[batch_size, 13 + len(mask)])
    tf_sp_weights = tf.SparseTensor(sp_indices, tf_sp_weight_vals, shape=[batch_size, 13 + len(mask)])
    tf_train_labels = tf.placeholder(tf.float32, shape=[batch_size])
    # tf_valid_dataset = tf.constant(tf.SparseTensor())
    # tf_test_dataset = tf.constant(test_dataset)

    weights = tf.Variable(tf.truncated_normal([X_dim, 1]))
    bias = tf.Variable(tf.zeros([1]))

    logits = tf.nn.embedding_lookup_sparse(weights, tf_sp_ids, tf_sp_weights, combiner='sum') + bias
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, tf_train_labels))

    optimizer = tf.train.GradientDescentOptimizer(_learning_rate).minimize(loss)

    train_pred = tf.sigmoid(logits)
    # valid_pred = tf.sigmoid(tf.matmul(tf_valid_dataset, weights, a_is_sparse=True) + bias)
    # test_pred = tf.sigmoid(tf.matmul(tf_test_dataset, weights, a_is_sparse=True) + bias)

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
        if step % 100 == 0:
            print 'loss as step %d: %f' % (step, l)
            print 'train-auc: %f%%\teval-auc: %f%%' % (
                roc_auc_score(batch_labels, pred),
                # roc_auc_score(valid_labels, valid_pred))
                0)
        if step == 1000:
            break
