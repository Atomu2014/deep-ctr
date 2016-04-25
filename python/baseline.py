import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, mean_squared_error, log_loss

from FM import FM
from FNN import FNN
from FPNN import FPNN
from LR import LR

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess_config = tf.ConfigProto(gpu_options=gpu_options)

train_path = '../data/nds.2.5.shuf.ind.100'
eval_path = '../data/test.unif.2.5.shuf.ind.100'
test_path = eval_path
nds_rate = 0.025
re_calibration = 'nds' not in eval_path
data_set = None

# nds = 0.1, ind = 10
# max_vals = [65535, 8000, 705, 249000, 7974, 14159, 946, 43970, 5448, 11, 471, 35070017, 8376]
# cat_sizes = np.array([631838, 25296, 15533, 7108, 19272, 3, 6753, 1302, 54, 527814, 147690, 110673,
#                       10, 2201, 9022, 62, 4, 945, 14, 639541, 357566, 600327, 118441, 10216, 77, 33])
# nds = 0.025, ind = 10
# max_vals = np.array([65535, 8000, 715, 376613, 7995, 1430, 2191, 25462, 5337, 8, 221, 3023897, 8072])
# cat_sizes = np.array([284414, 22289, 15249, 7022, 18956, 3, 6730, 1286, 50, 253234, 93011, 81371,
#                       10, 2187, 8396, 61, 4, 932, 14, 286828, 190408, 274540, 79781, 9690, 70, 33])
# nds = 0.025, ind = 100
max_vals = [65535, 8000, 715, 376613, 7995, 1430, 2191, 25462, 5337, 8, 221, 3023897, 8072]
cat_sizes = np.array(
    [31088, 13592, 13296, 6690, 17915, 3, 6497, 1217, 35, 34870, 24267, 32179, 10, 1959, 6076, 55, 4, 900, 14,
     30404, 34948, 32786, 22420, 8392, 50, 33])
cat_sizes += 1
offsets = [13 + sum(cat_sizes[:i]) for i in range(len(cat_sizes))]
X_dim = 13 + np.sum(cat_sizes)
# num of fields
X_feas = 13 + len(cat_sizes)

print 'max_vals:', max_vals
print 'cat_sizes (including \'other\'):', cat_sizes

mode = 'train'
# 'LR', 'FMxxx', 'FNN'
algo = 'LR'
tag = (time.strftime('%c') + ' ' + algo).replace(' ', '_')
log_path = '../log/%s' % tag
model_path = '../model/%s.pickle' % tag

print log_path, model_path

buffer_size = 1000000
eval_size = 10000
skip_window = 1
smooth_window = 100
stop_window = 10

seeds_pool = [0x0123, 0x4567, 0x3210, 0x7654, 0x89AB, 0xCDEF, 0xBA98, 0xFEDC, 0x0123, 0x4567, 0x3210, 0x7654, 0x89AB,
              0xCDEF, 0xBA98, 0xFEDC]

if 'LR' in algo:
    batch_size = 1
    test_batch_size = 1000
    epoch = 10000
    _rch_argv = [X_dim, X_feas]
    _min_val = -0.01
    _init_argv = ['uniform', _min_val, -1 * _min_val, seeds_pool[4:5], None]
    _ptmzr_argv = ['ftrl', 1e-3]
    _reg_argv = [1e-4]
elif 'FM' in algo:
    rank = int(algo[2:])
    batch_size = 1
    epoch = 100
    _rch_argv = [X_dim, X_feas, rank]
    _min_val = -1e-2
    _init_argv = ['uniform', _min_val, -1 * _min_val, seeds_pool[2:4], None]
    _ptmzr_argv = ['ftrl', 2e-3]
    _reg_argv = [1e-2]
elif 'FNN' in algo:
    rank = int(algo[3:])
    batch_size = 1
    epoch = 100
    _rch_argv = [X_dim, X_feas, rank, 400, 400, 'tanh']
    _min_val = -1e-2
    _init_argv = ['uniform', _min_val, -1 * _min_val, seeds_pool[4:9], None]
    _ptmzr_argv = ['adam', 1e-4, 1e-8]
    _reg_argv = [1e-3, 0.5]
elif 'FPNN' in algo:
    rank = int(algo[4:])
    batch_size = 1
    eval_batch_size = 20
    epoch = 1000
    _rch_argv = [X_dim, X_feas, rank, 800, 400, 'tanh']
    _min_val = -1e-2
    _init_argv = ['uniform', _min_val, -1 * _min_val, seeds_pool[4:9], None]
    _ptmzr_argv = ['adam', 1e-4, 1e-8]
    _reg_argv = [1e-3, 0.5]
else:
    exit(0)

ckpt = 10 * epoch
least_step = 100 * epoch
header = 'train_path: %s, eval_path: %s, tag: %s, nds_rate: %g, re_calibration: %s\n' \
         'batch_size: %d, epoch: %d, eval_size: %d, ckpt: %d, least_step: %d, skip_window: %d, smooth_window: %d, stop_window: %d' % \
         (train_path, eval_path, tag, nds_rate, str(re_calibration), batch_size, epoch, eval_size, ckpt, least_step,
          skip_window, smooth_window, stop_window)


def write_log(_line, echo=False):
    global mode
    if mode == 'test':
        return

    with open(log_path, 'a') as log_in:
        log_in.write(_line + '\n')
        if echo:
            print _line


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
                _rows.extend([row_num] * len(_f))
                _cols.extend(_f)
                _vals.extend(_x)
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


def watch_train(step, batch_loss, batch_labels, batch_preds, eval_preds, eval_labels):
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
    eval_ntrp = np.float32(log_loss(eval_labels, eval_preds))
    batch_ntrp = np.float32(log_loss(batch_labels, batch_preds))
    log = '%d\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t' % \
          (step, batch_ntrp, eval_ntrp, batch_auc, eval_auc, batch_rmse, eval_rmse, batch_loss)
    write_log(log)
    return {'batch_loss': batch_loss, 'batch_entropy': batch_ntrp, 'eval_entropy': eval_ntrp, 'batch_auc': batch_auc,
            'eval_auc': eval_auc, 'batch_rmse': batch_rmse, 'eval_rmse': eval_rmse}


def early_stop(step, errs):
    if step > least_step:
        skip_auc = errs[::skip_window]
        smooth_auc = np.array(skip_auc[smooth_window - 1:])
        for i in range(smooth_window - 1):
            smooth_auc += skip_auc[i:(i - smooth_window + 1)]
        smooth_auc /= smooth_window
        if len(smooth_auc) < stop_window:
            return False
        smooth_error = smooth_auc[stop_window - 1:] - smooth_auc[:1 - stop_window]
        if smooth_error[-1] > 0:
            print 'early stop at step %d' % step
            print 'smoothed error', str(smooth_error)
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

    if 'LR' in algo:
        model = LR(batch_size, _rch_argv, _init_argv, _ptmzr_argv, _reg_argv, 'train', eval_size, eval_cols, eval_wts)
    elif 'FM' in algo:
        model = FM(batch_size, _rch_argv, _init_argv, _ptmzr_argv, _reg_argv, 'train', eval_size, eval_cols, eval_wts)
    elif 'FNN' in algo:
        eval_cols = eval_cols.reshape((eval_size, X_feas))
        eval_wts = np.float32(eval_wts.reshape((eval_size, X_feas)))
        model = FNN(cat_sizes, offsets, batch_size, _rch_argv, _init_argv, _ptmzr_argv, _reg_argv, 'train', eval_size,
                    eval_cols, eval_wts)
    elif 'FPNN' in algo:
        eval_cols = eval_cols.reshape((eval_size, X_feas))
        eval_wts = np.float32(eval_wts.reshape((eval_size, X_feas)))
        model = FPNN(cat_sizes, offsets, batch_size, _rch_argv, _init_argv, _ptmzr_argv, _reg_argv, 'train',
                     eval_batch_size)

    write_log(model.log, echo=True)

    with tf.Session(graph=model.graph, config=sess_config) as sess:
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
                elif 'FPNN' in algo:
                    _cols = _cols.reshape((batch_size, X_feas))
                    _vals = _vals.reshape((batch_size, X_feas))
                    feed_dict = {model.v_wt_hldr: _vals[:, :13], model.c_id_hldr: _cols[:, 13:] - offsets,
                                 model.c_wt_hldr: _vals[:, 13:], model.lbl_hldr: _labels}

                _, l, p = sess.run([model.ptmzr, model.loss, model.train_preds], feed_dict=feed_dict)
                batch_preds.extend(_x[0] for _x in p)
                batch_labels.extend(_labels)
                if step % epoch == 0:
                    print 'step: %d\ttime: %d' % (step, time.time() - start_time)
                    start_time = time.time()
                    if 'FPNN' in algo:
                        eval_preds = []
                        for _i in range(eval_size / eval_batch_size):
                            eval_preds.append(model.eval_preds.eval(feed_dict={
                                model.eval_id_hldr: eval_cols[_i * eval_batch_size:(_i + 1) * eval_batch_size, :],
                                model.eval_wts_hldr: eval_wts[_i * eval_batch_size:(_i + 1) * eval_batch_size, :]}))
                        eval_preds = np.reshape(eval_preds, (-1,))
                    else:
                        eval_preds = model.eval_preds.eval()
                    if re_calibration:
                        eval_preds /= eval_preds + (1 - eval_preds) / nds_rate
                    metrics = watch_train(step, l, batch_labels, batch_preds, eval_preds, eval_labels)
                    print metrics
                    err_rcds.append(metrics['eval_entropy'])
                    err_rcds = err_rcds[-2 * skip_window * (stop_window + smooth_window):]
                    batch_preds = []
                    batch_labels = []
                    if step % ckpt == 0:
                        model.dump(model_path + '_%d' % step)
                    if early_stop(step, err_rcds):
                        return


def test():
    if 'LR' in algo:
        model = LR(test_batch_size, _rch_argv, _init_argv, None, None, 'test', None, None, None)
    elif 'FM' in algo:
        model = FM(batch_size, _rch_argv, _init_argv, _ptmzr_argv, _reg_argv, 'test', eval_size, None, None)
    elif 'FNN' in algo:
        model = FNN(cat_sizes, offsets, batch_size, _rch_argv, _init_argv, _ptmzr_argv, _reg_argv, 'test', eval_size,
                    None, None)
    elif 'FPNN' in algo:
        model = FPNN(cat_sizes, offsets, batch_size, _rch_argv, _init_argv, _ptmzr_argv, _reg_argv, 'test', None)

    with tf.Session(graph=model.graph, config=sess_config) as sess:
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
                step += 1
                _labels = labels[_i * test_batch_size: (_i + 1) * test_batch_size]
                _cols = cols[_i * test_batch_size * X_feas: (_i + 1) * test_batch_size * X_feas]
                _vals = vals[_i * test_batch_size * X_feas: (_i + 1) * test_batch_size * X_feas]

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
                elif 'FPNN' in algo:
                    _cols = _cols.reshape((batch_size, X_feas))
                    _vals = _vals.reshape((batch_size, X_feas))
                    feed_dict = {model.v_wt_hldr: _vals[:, :13], model.c_id_hldr: _cols[:, 13:] - offsets,
                                 model.c_wt_hldr: _vals[:, 13:], model.lbl_hldr: _labels}

                p = model.test_preds.eval(feed_dict=feed_dict)
                p /= p + (1 - p) / nds_rate

                test_preds.extend(_x[0] for _x in p)
                test_labels.extend(_labels)
                if step % 100000 == 0:
                    print 'test-auc: %g' % roc_auc_score(test_labels, test_preds)
                    print 'test-rmse: %g' % np.sqrt(mean_squared_error(test_labels, test_preds))
                    print 'test-log-loss: %g' % log_loss(test_labels, test_preds)
                    print time.time() - start_time
                    start_time = time.time()

        if len(labels) < buffer_size:
            print 'test-auc: %g' % roc_auc_score(test_labels, test_preds)
            print 'test-rmse: %g' % np.sqrt(mean_squared_error(test_labels, test_preds))
            print 'test-log-loss: %g' % log_loss(test_labels, test_preds)
            exit(0)


if __name__ == '__main__':
    if mode == 'train':
        train()
    else:
        test()
