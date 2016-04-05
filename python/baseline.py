import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, mean_squared_error, log_loss

from FM import FM
from FNN import FNN
from LR import LR

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess_config = tf.ConfigProto(gpu_options=gpu_options)

train_path = '../data/nds.2.5.shuf.ind.10'
eval_path = '../data/test.nds.2.5.shuf.ind.10'
nds_rate = 0.025
re_calibration = 'nds' not in '../data/test.nds.2.5.shuf.ind.10'
train_set = None

# nds = 0.1, ind = 10
# max_vals = [65535, 8000, 705, 249000, 7974, 14159, 946, 43970, 5448, 11, 471, 35070017, 8376]
# cat_sizes = np.array([631838, 25296, 15533, 7108, 19272, 3, 6753, 1302, 54, 527814, 147690, 110673,
#                       10, 2201, 9022, 62, 4, 945, 14, 639541, 357566, 600327, 118441, 10216, 77, 33])
# nds = 0.025, ind = 10
max_vals = np.array([65535, 8000, 715, 376613, 7995, 1430, 2191, 25462, 5337, 8, 221, 3023897, 8072])
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
algo = 'FM2'
tag = (time.strftime('%c') + ' ' + algo).replace(' ', '_')
if 'FM' in algo:
    rank = int(algo[2:])
if 'FNN' in algo:
    rank = int(algo[3:])
    fnn_init_path = ''
log_path = '../log/%s' % tag
model_path = '../model/%s.pickle' % tag

print log_path, model_path

buffer_size = 1000000
eval_size = 10000
skip_window = 1
smooth_window = 100
stop_window = 10

if 'LR' in algo:
    batch_size = 1
    epoch = 1000
    _learning_rate = 1e-4
    _min_val = -1e-2
    _alpha = 0
    _lambda = 0.001
    _epsilon = 1e-8
    _stddev = 0.001
elif 'FM' in algo:
    batch_size = 1
    epoch = 1000
    _learning_rate = 1e-8
    _min_val = -1e-3
    _alpha = 0
    _lambda = 0.01
    _epsilon = 1e-8
    _stddev = 0.001
elif 'FNN' in algo:
    batch_size = 10
    epoch = 10
    _learning_rate = 1e-4
    _min_val = -1e-2
    _alpha = 0
    _lambda = 0
    _epsilon = 1e-8
    _stddev = 0.001
else:
    batch_size = 10
    epoch = 1000
    _learning_rate = 1e-5
    _min_val = -0.001
    _alpha = 0
    _lambda = 0.001
    _epsilon = 1e-8
    _stddev = 0.001

_keep_prob = 0.5
ckpt = 10 * epoch
least_step = 1000 * epoch
# 'normal', 't-normal', 'uniform'(default)
_init_method = 'uniform'
_max_val = -1 * _min_val
seeds_pool = [0x0123, 0x4567, 0x3210, 0x7654, 0x89AB, 0xCDEF, 0xBA98, 0xFEDC]
# _seeds = seeds_pool[0:2]
_seeds = seeds_pool[2:]
# _seeds = seeds_pool[4:6]
# _seeds = seeds_pool[6:8]

headers = ['train_path: %s, eval_path: %s, tag: %s, nds_rate: %g, re_calibration: %s' % (
    train_path, eval_path, tag, nds_rate, str(re_calibration)),
           'batch_size: %d, epoch: %d, buffer_size: %d, eval_size: %d, ckpt: %d, least_step: %d, skip_window: %d, smooth_window: %d, stop_window: %d' % (
               batch_size, epoch, buffer_size, eval_size, ckpt, least_step, skip_window, smooth_window, stop_window),
           'learning_rate: %g, alpha:%g, lambda: %g, epsilon:%g, keep_prob: %g' % (
               _learning_rate, _alpha, _lambda, _epsilon, _keep_prob),
           'init_method: %s, stddev: %g, interval: [%g, %g], seeds: %s' % (
               _init_method, _stddev, _min_val, _max_val, str(_seeds))]


def write_log(argvs, erase=False):
    if erase:
        mode = 'w'
    else:
        mode = 'a'
    with open(log_path, mode) as log_in:
        log_in.write('\t'.join([str(_x) for _x in argvs]) + '\n')


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
    global train_set
    if train_set is None:
        train_set = open(train_path, 'rb')
    _labels = []
    _cols = []
    _vals = []
    _rows = []
    row_num = row_start
    for _i in range(size):
        try:
            _line = next(train_set)
            if len(_line.strip()):
                _y, _f, _x = get_fxy(_line)
                _rows.extend([row_num] * len(_f))
                _cols.extend(_f)
                _vals.extend(_x)
                _labels.append(_y)
                row_num += 1
            else:
                break
        except StopIteration as e:
            print e
            train_set = None
            break

    return _labels, _rows, _cols, _vals


def get_batch_xy(size):
    _labels, _rows, _cols, _vals = get_batch_sparse_tensor(size)
    if len(_labels) == size:
        return np.array(_labels), np.array(_rows), np.array(_cols), np.array(_vals)

    _l, _r, _c, _v = get_batch_sparse_tensor(size - len(_labels))
    _labels.extend(_l)
    _rows.extend(_r)
    _cols.extend(_c)
    _vals.extend(_v)
    return np.array(_labels), np.array(_rows), np.array(_cols), np.array(_vals)


eval_labels = []
eval_cols = []
eval_wts = []
eval_cnt = 0
with open(eval_path, 'r') as eval_set:
    buf = []
    for line in eval_set:
        buf.append(line)
        eval_cnt += 1
        if eval_cnt == 10 * eval_size:
            break
    np.random.shuffle(buf)
    buf = buf[:eval_size]
    for line in buf:
        y, f, x = get_fxy(line)
        eval_cols.extend(f)
        eval_wts.extend(x)
        eval_labels.append(y)
eval_cols = np.array(eval_cols)
eval_wts = np.array(eval_wts)
eval_labels = np.array(eval_labels)

sp_eval_inds = []
for i in range(eval_size):
    for j in range(X_feas):
        sp_eval_inds.append([i, j])
sp_eval_inds = np.array(sp_eval_inds)

sp_train_inds = []
for i in range(batch_size):
    for j in range(X_feas):
        sp_train_inds.append([i, j])
sp_train_inds = np.array(sp_train_inds)


def watch_train(step, batch_loss, batch_labels, batch_preds, eval_preds):
    try:
        batch_auc = np.float32(roc_auc_score(batch_labels, batch_preds))
    except ValueError:
        batch_auc = -1
    eval_auc = np.float32(roc_auc_score(eval_labels, eval_preds))
    batch_rmse = np.float32(np.sqrt(mean_squared_error(batch_labels, batch_preds)))
    eval_rmse = np.float32(np.sqrt(mean_squared_error(eval_labels, eval_preds)))
    eval_ntrp = np.float32(log_loss(eval_labels, eval_preds))
    batch_ntrp = np.float32(log_loss(batch_labels, batch_preds))
    write_log([step, batch_ntrp, eval_ntrp, batch_auc, eval_auc, batch_rmse, eval_rmse, batch_loss])
    return {'batch_loss': batch_loss, 'batch_entropy': batch_ntrp, 'eval_entropy': eval_ntrp, 'batch_auc': batch_auc,
            'eval_auc': eval_auc,
            'batch_rmse': batch_rmse, 'eval_rmse': eval_rmse}


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


def train():
    global eval_cols, eval_wts, eval_labels

    if 'LR' in algo:
        model = LR(batch_size, eval_size, X_dim, X_feas, sp_train_inds, sp_eval_inds, eval_cols, eval_wts, _min_val,
                   _max_val, _seeds, _learning_rate, _alpha, _lambda, _epsilon)
    elif 'FM' in algo:
        model = FM(batch_size, eval_size, X_dim, X_feas, sp_train_inds, sp_eval_inds, eval_cols, eval_wts, rank,
                   _min_val, _max_val, _seeds, _learning_rate, _lambda, _epsilon)
    elif 'FNN' in algo:
        eval_cols = eval_cols.reshape((eval_size, X_feas))
        eval_wts = np.float32(eval_wts.reshape((eval_size, X_feas)))
        model = FNN(cat_sizes, offsets, batch_size, eval_size, X_dim, X_feas, eval_cols, eval_wts, rank, _min_val,
                    _max_val, _seeds, _learning_rate, _alpha, _lambda, _epsilon, _keep_prob)
    else:
        return

    with tf.Session(graph=model.graph, config=sess_config) as sess:
        tf.initialize_all_variables().run()
        print 'model initialized'
        start_time = time.time()
        step = 0
        batch_preds = []
        batch_labels = []
        auc_track = []
        while True:
            labels, _, cols, vals = get_batch_xy(buffer_size)
            for _i in range(labels.shape[0] / batch_size):
                step += 1
                _labels = labels[_i * batch_size: (_i + 1) * batch_size]
                _cols = cols[_i * batch_size * X_feas: (_i + 1) * batch_size * X_feas]
                _vals = vals[_i * batch_size * X_feas: (_i + 1) * batch_size * X_feas]

                if 'LR' in algo:
                    feed_dict = {model.sp_id_hldr: _cols, model.sp_wt_hldr: _vals, model.lbl_hldr: _labels}
                elif 'FM' in algo:
                    _vals2 = _vals ** 2
                    feed_dict = {model.sp_id_hldr: _cols, model.sp_wt_hldr: _vals, model.sp_wt2_hldr: _vals2,
                                 model.lbl_hldr: _labels}
                elif 'FNN' in algo:
                    _cols = _cols.reshape((batch_size, X_feas))
                    _vals = _vals.reshape((batch_size, X_feas))
                    feed_dict = {model.v_wt_hldr: _vals[:, :13], model.c_id_hldr: _cols[:, 13:] - offsets,
                                 model.c_wt_hldr: _vals[:, 13:], model.lbl_hldr: _labels}

                _, l, p = sess.run([model.ptmzr, model.loss, model.train_preds], feed_dict=feed_dict)
                batch_preds.extend(_x[0] for _x in p)
                batch_labels.extend(_labels)
                if step % epoch == 0:
                    print 'step: %d\ttime: %d' % (step, time.time() - start_time)
                    print model.eval_logits.eval()
                    start_time = time.time()
                    metrics = watch_train(step, l, batch_labels, batch_preds, model.eval_preds.eval())
                    print metrics
                    auc_track.append(metrics['eval_auc'])
                    auc_track = auc_track[-2 * skip_window * (stop_window + smooth_window):]
                    batch_preds = []
                    batch_labels = []
                    if step % ckpt == 0:
                        model.dump(model_path)
                    if early_stop(step, auc_track):
                        return


if __name__ == '__main__':
    train()
