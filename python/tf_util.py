import cPickle as pickle

import tensorflow as tf


def builf_optimizer(_ptmzr_argv, loss):
    _ptmzr = _ptmzr_argv[0]
    if _ptmzr == 'adam':
        _learning_rate, _epsilon = _ptmzr_argv[1:]
        ptmzr = tf.train.AdamOptimizer(learning_rate=_learning_rate, epsilon=_epsilon).minimize(loss)
        log = 'optimizer: %s, learning rate: %g, epsilon: %g' % (_ptmzr, _learning_rate, _epsilon)
    elif _ptmzr == 'ftrl':
        _learning_rate = _ptmzr_argv[1]
        ptmzr = tf.train.FtrlOptimizer(learning_rate=_learning_rate).minimize(loss)
        log = 'optimizer: %s, learning rate: %g' % (_ptmzr, _learning_rate)
    else:
        _learning_rate = _ptmzr_argv[1]
        ptmzr = tf.train.GradientDescentOptimizer(learning_rate=_learning_rate).minimize(loss)
        log = 'optimizer: %s, learning rate: %g' % (_ptmzr, _learning_rate)
    return ptmzr, log


def init_var_map(_init_argv, vars):
    _init_path = _init_argv[-1]
    if _init_path:
        var_map = pickle.load(open(_init_path, 'rb'))
        log = 'init model from: %s, ' % _init_path
    else:
        var_map = {}
        log = 'random init, '

    _init_method = _init_argv[0]
    if _init_method == 'normal':
        _mean, _stddev, _seeds = _init_argv[1:-1]
        log += 'init method: %s(mean=%g, stddev=%g), seeds: %s\n' % (_init_method, _mean, _stddev, str(_seeds))
        _j = 0
        for _i in range(len(vars)):
            key, shape, action = vars[_i]
            if key not in var_map.keys():
                if action == 'random':
                    var_map[key] = tf.random_normal(shape, _mean, _stddev, seed=_seeds[_j])
                    _j += 1
                else:
                    var_map[key] = tf.zeros(shape)
    else:
        _min_val, _max_val, _seeds = _init_argv[1:-1]
        log += 'init method: %s(minval=%g, maxval=%g), seeds: %s\n' % (
            _init_method, _min_val, _max_val, str(_seeds))
        _j = 0
        for _i in range(len(vars)):
            key, shape, action = vars[_i]
            if key not in var_map.keys():
                if action == 'random':
                    var_map[key] = tf.random_uniform(shape, _min_val, _max_val, seed=_seeds[_j])
                    _j += 1
                else:
                    var_map[key] = tf.zeros(shape)

    return var_map, log
