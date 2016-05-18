import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, mean_squared_error, log_loss

from FM import FM
from FNN import FNN
from FNN_IP_L3 import FNN_IP_L3
from FNN_IP_L5 import FNN_IP_L5
from FNN_IP_L7 import FNN_IP_L7
from FNN_OP_L3 import FNN_OP_L3
from LR import LR

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
# sess_config = tf.ConfigProto(gpu_options=gpu_options)

train_path = '../data/nds.2.5.shuf.ind.10'
eval_path = '../data/test.unif.2.5.shuf.ind.10'
test_path = eval_path
nds_rate = 0.025
re_calibration = 'nds' not in eval_path
data_set = None

# nds = 0.1, ind = 10
# max_vals = [65535, 8000, 705, 249000, 7974, 14159, 946, 43970, 5448, 11, 471, 35070017, 8376]
# cat_sizes = np.array([631838, 25296, 15533, 7108, 19272, 3, 6753, 1302, 54, 527814, 147690, 110673,
#                       10, 2201, 9022, 62, 4, 945, 14, 639541, 357566, 600327, 118441, 10216, 77, 33])
# nds = 0.025, ind = 10
max_vals = np.array([65535, 8000, 715, 376613, 7995, 1430, 2191, 25462, 5337, 8, 221, 3023897, 8072])
cat_sizes = np.array([284414, 22289, 15249, 7022, 18956, 3, 6730, 1286, 50, 253234, 93011, 81371,
                      10, 2187, 8396, 61, 4, 932, 14, 286828, 190408, 274540, 79781, 9690, 70, 33])
# nds = 0.025, ind = 100
# max_vals = [65535, 8000, 715, 376613, 7995, 1430, 2191, 25462, 5337, 8, 221, 3023897, 8072]
# cat_sizes = np.array(
#     [31088, 13592, 13296, 6690, 17915, 3, 6497, 1217, 35, 34870, 24267, 32179, 10, 1959, 6076, 55, 4, 900, 14,
#      30404, 34948, 32786, 22420, 8392, 50, 33])

cat_sizes += 1
offsets = [13 + sum(cat_sizes[:i]) for i in range(len(cat_sizes))]
X_dim = 13 + np.sum(cat_sizes)
# num of fields
X_feas = 13 + len(cat_sizes)

print 'max_vals:', max_vals
print 'cat_sizes (including \'other\'):', cat_sizes

mode = 'train'
# 'LR', 'FMxxx', 'FNN'
algo = 'FNN_IP_L3_10'
tag = (time.strftime('%c') + ' ' + algo).replace(' ', '_')
log_path = '../log/%s' % tag
model_path = '../model/%s.pickle' % tag

print log_path, model_path

buffer_size = 1000000
eval_buf_size = 100000
skip_window = 1
smooth_window = 10
stop_window = 10

seeds_pool = [0x0123, 0x4567, 0x3210, 0x7654, 0x89AB, 0xCDEF, 0xBA98, 0xFEDC, 0x0123, 0x4567, 0x3210, 0x7654, 0x89AB,
              0xCDEF, 0xBA98, 0xFEDC]

if 'LR' in algo:
    batch_size = 10
    eval_size = 10000
    test_batch_size = 1000
    epoch = 100000
    _rch_argv = [X_dim, X_feas]
    _min_val = -0.01
    _init_argv = ['uniform', _min_val, -1 * _min_val, seeds_pool[4:5],
                  '../model/Sun_Apr_24_15:03:10_2016_LR.pickle_21200000']
    _ptmzr_argv = ['ftrl', 1e-3]
    _reg_argv = [1e-4]
elif 'FM' in algo:
    rank = int(algo[2:])
    batch_size = 100
    eval_size = 100
    test_batch_size = 1000
    epoch = 100000
    _rch_argv = [X_dim, X_feas, rank]
    _min_val = -1e-2
    _init_argv = ['uniform', _min_val, -1 * _min_val, seeds_pool[2:4],
                  # None]
                  # FM50
                  # '../model/Sat_May_14_17:41:41_2016_FM50.pickle_22000']
                  # FM10
                  # '../model/Wed_May__4_19_53_08_2016_FM10.pickle_7000000']
                  # FM100
                  '../model/Sat_May_14_18:43:20_2016_FM100.pickle_30000']
    _ptmzr_argv = ['adam', 1e-4, 1e-8, 'sum']
    _reg_argv = [1e-3]
elif 'FNN_IP_L3_' in algo:
    rank = int(algo[10:])
    batch_size = 10
    eval_size = 50
    test_batch_size = 50
    epoch = 10000
    _rch_argv = [X_dim, X_feas, rank, 800, 400, 200, 'relu']
    _min_val = -1e-2
    _init_argv = ['uniform', _min_val, -1 * _min_val, seeds_pool[4:10],
                  # '../model/Thu_May_12_16:35:29_2016_FPNN_H3_10.pickle_194000']
                  None]
    _ptmzr_argv = ['adam', 1e-4, 1e-8, 'sum']
    _reg_argv = [0.5]
elif 'FNN_OP_L3_' in algo:
    rank = int(algo[10:])
    batch_size = 10
    eval_size = 50
    test_batch_size = 50
    epoch = 10000
    _rch_argv = [X_dim, X_feas, rank, 800, 400, 200, 'relu']
    _min_val = -1e-2
    _init_argv = ['uniform', _min_val, -1 * _min_val, seeds_pool[4:10],
                  None]
    _ptmzr_argv = ['adam', 1e-4, 1e-8, 'sum']
    _reg_argv = [0.5]
elif 'FNN_IP_L5_' in algo:
    rank = int(algo[10:])
    batch_size = 50
    eval_size = 10
    test_batch_size = 50
    epoch = 2000
    _rch_argv = [X_dim, X_feas, rank, 1000, 800, 600, 400, 200, 'relu']
    _min_val = -1e-2
    _init_argv = ['uniform', _min_val, -1 * _min_val, seeds_pool[4:12],
                  '../model/Wed_May_18_00:04:27_2016_FNN_IP_L5_10.pickle_32500']
    _ptmzr_argv = ['adam', 1e-4, 1e-8, 'sum']
    _reg_argv = [0.5]
elif 'FNN_IP_L7_' in algo:
    rank = int(algo[10:])
    batch_size = 10
    eval_size = 50
    test_batch_size = 50
    epoch = 10000
    _rch_argv = [X_dim, X_feas, rank, 1000, 800, 600, 400, 200, 100, 50, 'relu']
    _min_val = -1e-2
    _init_argv = ['uniform', _min_val, -1 * _min_val, seeds_pool[4:14],
                  None]
    _ptmzr_argv = ['adam', 1e-4, 1e-8, 'sum']
    _reg_argv = [0.5]
elif 'FNN' in algo:
    rank = int(algo[3:])
    batch_size = 100
    eval_size = 1000
    test_batch_size = 1000
    epoch = 1000
    _rch_argv = [X_dim, X_feas, rank, 400, 400, 'relu']
    _min_val = -1e-2
    _init_argv = ['uniform', _min_val, -1 * _min_val, seeds_pool[4:9],
                  # tanh
                  # '../model/Thu_May_12_14:17:57_2016_FNN10.pickle_100000']
                  # relu
                  # '../model/Mon_May_16_14:05:17_2016_FNN10.pickle_142000']
                  None]
    _ptmzr_argv = ['adam', 1e-4, 1e-8, 'sum']
    _reg_argv = [0.5]
else:
    exit(0)

ckpt = epoch
least_step = 10 * epoch
header = 'train_path: %s, eval_path: %s, tag: %s, nds_rate: %g, re_calibration: %s\n' \
         'batch_size: %d, epoch: %d, eval_size: %d, ckpt: %d, least_step: %d, skip_window: %d, smooth_window: %d, stop_window: %d' % \
         (train_path, eval_path, tag, nds_rate, str(re_calibration), batch_size, epoch, eval_buf_size, ckpt, least_step,
          skip_window, smooth_window, stop_window)


def write_log(_line, echo=False):
    global mode
    if mode == 'test':
        return

    with open(log_path, 'a') as log_in:
        log_in.write(_line + '\n')
        if echo:
            print _line


write_log(header, True)


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
    global data_set
    if data_set is None:
        if mode == 'train':
            data_set = open(train_path, 'rb')
        else:
            data_set = open(test_path, 'rb')
    _labels = []
    _cols = []
    _vals = []
    _rows = []
    row_num = row_start
    for _i in range(size):
        try:
            _line = next(data_set)
            if len(_line.strip()):
                _y, _f, _x = get_fxy(_line)
                _rows.append([row_num] * len(_f))
                _cols.append(_f)
                _vals.append(_x)
                _labels.append(_y)
                row_num += 1
            else:
                break
        except StopIteration as e:
            print e
            data_set = None
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


def watch_train(step, batch_labels, batch_preds, eval_preds, eval_labels):
    try:
        batch_auc = np.float32(roc_auc_score(batch_labels, batch_preds))
    except ValueError:
        batch_auc = -1
    try:
        eval_auc = np.float32(roc_auc_score(eval_labels, eval_preds))
    except ValueError:
        eval_auc = -1
    batch_rmse = np.float32(np.sqrt(mean_squared_error(batch_labels, batch_preds)))
    eval_rmse = np.float32(np.sqrt(mean_squared_error(eval_labels, eval_preds)))
    log = '%d\t%g\t%g\t%g\t%g\t' % (step, batch_auc, eval_auc, batch_rmse, eval_rmse)
    write_log(log)
    return {'batch_auc': batch_auc, 'eval_auc': eval_auc, 'batch_rmse': batch_rmse, 'eval_rmse': eval_rmse}


def early_stop(step, errs, metric='auc'):
    if step > least_step:
        skip_metric = errs[::skip_window]
        smooth_metric = np.array(skip_metric[smooth_window - 1:])
        for i in range(smooth_window - 1):
            smooth_metric += skip_metric[i:(i - smooth_window + 1)]
        smooth_metric /= smooth_window
        if len(smooth_metric) < stop_window:
            return False
        smooth_error = smooth_metric[stop_window - 1:] - smooth_metric[:1 - stop_window]
        if metric == 'rmse' and smooth_error[-1] > 0:
            print 'early stop at step %d' % step
            print 'smoothed rmse error', str(smooth_error)
            return True
        elif metric == 'auc' and smooth_error[-1] < 0:
            print 'early stop at step %d' % step
            print 'smoothed auc error', str(smooth_error)
            return True
        return False
    return False


def train():
    eval_labels = []
    eval_cols = []
    eval_wts = []
    eval_cnt = 0
    with open(eval_path, 'r') as eval_set:
        buf = []
        for line in eval_set:
            buf.append(line)
            eval_cnt += 1
            if eval_cnt == eval_buf_size:
                break
        np.random.shuffle(buf)
        for line in buf:
            y, f, x = get_fxy(line)
            eval_cols.append(f)
            eval_wts.append(x)
            eval_labels.append(y)
    eval_cols = np.array(eval_cols)
    eval_wts = np.float32(np.array(eval_wts))
    eval_labels = np.array(eval_labels)

    if 'LR' in algo:
        model = LR(batch_size, _rch_argv, _init_argv, _ptmzr_argv, _reg_argv, 'train', eval_size)
    elif 'FM' in algo:
        model = FM(batch_size, _rch_argv, _init_argv, _ptmzr_argv, _reg_argv, 'train', eval_size)
    elif 'FNN_IP_L3' in algo:
        model = FNN_IP_L3(cat_sizes, offsets, batch_size, _rch_argv, _init_argv, _ptmzr_argv, _reg_argv, 'train',
                          eval_size)
    elif 'FNN_OP_L3' in algo:
        model = FNN_OP_L3(cat_sizes, offsets, batch_size, _rch_argv, _init_argv, _ptmzr_argv, _reg_argv, 'train',
                          eval_size)
    elif 'FNN_IP_L5' in algo:
        model = FNN_IP_L5(cat_sizes, offsets, batch_size, _rch_argv, _init_argv, _ptmzr_argv, _reg_argv, 'train',
                          eval_size)
    elif 'FNN_IP_L7' in algo:
        model = FNN_IP_L7(cat_sizes, offsets, batch_size, _rch_argv, _init_argv, _ptmzr_argv, _reg_argv, 'train',
                          eval_size)
    elif 'FNN' in algo:
        model = FNN(cat_sizes, offsets, batch_size, _rch_argv, _init_argv, _ptmzr_argv, _reg_argv, 'train', eval_size)

    write_log(model.log, echo=True)

    # with tf.Session(graph=model.graph, config=sess_config) as sess:
    with tf.Session(graph=model.graph) as sess:
        tf.initialize_all_variables().run()
        print 'model initialized'
        start_time = time.time()
        step = 0
        batch_preds = []
        batch_labels = []
        err_rcds = []
        while True:
            labels, _, cols, vals = get_batch_xy(buffer_size)
            for _i in range(labels.shape[0] / batch_size):
                step += 1
                _labels = labels[_i * batch_size: (_i + 1) * batch_size]
                _cols = cols[_i * batch_size: (_i + 1) * batch_size, :]
                _vals = vals[_i * batch_size: (_i + 1) * batch_size, :]

                if 'LR' in algo or 'FM' in algo:
                    feed_dict = {model.sp_id_hldr: _cols.flatten(), model.sp_wt_hldr: _vals.flatten(),
                                 model.lbl_hldr: _labels}
                elif 'FNN' in algo:
                    feed_dict = {model.v_wt_hldr: _vals[:, :13], model.c_id_hldr: _cols[:, 13:] - offsets,
                                 model.c_wt_hldr: _vals[:, 13:], model.lbl_hldr: _labels}

                _, l, p = sess.run([model.ptmzr, model.loss, model.train_preds], feed_dict=feed_dict)
                batch_preds.extend(p)
                batch_labels.extend(_labels)
                if step % epoch == 0:
                    print 'step: %d\tloss: %g\ttime: %d' % (step * batch_size, l, time.time() - start_time)
                    start_time = time.time()
                    eval_preds = []
                    for _i in range(eval_buf_size / eval_size):
                        eval_inds = eval_cols[_i * eval_size:(_i + 1) * eval_size]
                        eval_vals = eval_wts[_i * eval_size:(_i + 1) * eval_size]
                        if 'LR' in algo or 'FM' in algo:
                            feed_dict = {model.eval_id_hldr: eval_inds.flatten(),
                                         model.eval_wts_hldr: eval_vals.flatten()}
                        elif 'FNN' in algo or 'FPNN' in algo:
                            feed_dict = {model.eval_id_hldr: eval_inds, model.eval_wts_hldr: eval_vals}
                        eval_preds.extend(model.eval_preds.eval(feed_dict=feed_dict))
                    eval_preds = np.array(eval_preds)
                    if re_calibration:
                        eval_preds /= eval_preds + (1 - eval_preds) / nds_rate
                    metrics = watch_train(step, batch_labels, batch_preds, eval_preds, eval_labels)
                    print metrics
                    err_rcds.append(metrics['eval_auc'])
                    err_rcds = err_rcds[-2 * skip_window * (stop_window + smooth_window):]
                    batch_preds = []
                    batch_labels = []
                    if step % ckpt == 0:
                        model.dump(model_path + '_%d' % step)
                    if early_stop(step, err_rcds):
                        return


def test():
    if 'LR' in algo:
        model = LR(test_batch_size, _rch_argv, _init_argv, None, None, 'test', None)
    elif 'FM' in algo:
        model = FM(test_batch_size, _rch_argv, _init_argv, None, None, 'test', None)
    elif 'FNN_IP_L3' in algo:
        model = FNN_IP_L3(cat_sizes, offsets, test_batch_size, _rch_argv, _init_argv, None, None, 'test', None)
    elif 'FNN_IP_L5' in algo:
        model = FNN_IP_L5(cat_sizes, offsets, test_batch_size, _rch_argv, _init_argv, None, None, 'test', None)
    elif 'FNN_IP_ L7' in algo:
        model = FNN_IP_L7(cat_sizes, offsets, test_batch_size, _rch_argv, _init_argv, None, None, 'test', None)
    elif 'FNN' in algo:
        model = FNN(cat_sizes, offsets, test_batch_size, _rch_argv, _init_argv, None, None, 'test', None)

    print 'testing model: %s' % _init_argv[-1]
    # with tf.Session(graph=model.graph, config=sess_config) as sess:
    with tf.Session(graph=model.graph) as sess:
        tf.initialize_all_variables().run()
        print 'model initialized'
        test_preds = []
        test_labels = []
        step = 0
        start_time = time.time()
        while True:
            labels, _, cols, vals = get_batch_sparse_tensor(buffer_size)
            labels, cols, vals = np.array(labels), np.array(cols), np.array(vals)
            for _i in range(labels.shape[0] / test_batch_size):
                step += test_batch_size
                _labels = labels[_i * test_batch_size: (_i + 1) * test_batch_size]
                _cols = cols[_i * test_batch_size: (_i + 1) * test_batch_size, :]
                _vals = vals[_i * test_batch_size: (_i + 1) * test_batch_size, :]

                if 'LR' in algo or 'FM' in algo:
                    feed_dict = {model.sp_id_hldr: _cols.flatten(), model.sp_wt_hldr: _vals.flatten(),
                                 model.lbl_hldr: _labels}
                elif 'FNN' in algo or 'FPNN' in algo:
                    feed_dict = {model.v_wt_hldr: _vals[:, :13], model.c_id_hldr: _cols[:, 13:] - offsets,
                                 model.c_wt_hldr: _vals[:, 13:], model.lbl_hldr: _labels}

                p = model.test_preds.eval(feed_dict=feed_dict)
                p /= p + (1 - p) / nds_rate

                test_preds.extend(p)
                test_labels.extend(_labels)
                if step % epoch == 0:
                    print 'test-auc: %g\trmse: %g\tlog-loss: %g' % (
                        roc_auc_score(test_labels, test_preds), np.sqrt(mean_squared_error(test_labels, test_preds)),
                        log_loss(test_labels, test_preds))
                    print 'step: %d\ttime: %g' % (step, time.time() - start_time)
                    start_time = time.time()

            if len(labels) < buffer_size:
                print 'test-auc: %g\trmse: %g\tlog-loss: %g' % (
                    roc_auc_score(test_labels, test_preds), np.sqrt(mean_squared_error(test_labels, test_preds)),
                    log_loss(test_labels, test_preds))
                exit(0)


if __name__ == '__main__':
    if mode == 'train':
        train()
    else:
        test()
