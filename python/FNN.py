import cPickle as pickle

import tensorflow as tf


class FNN:
    def __init__(self, cat_sizes, offsets, batch_size, eval_size, X_dim, X_feas, eval_cols, eval_wts, rank, _min_val,
                 _max_val, _seeds, _learning_rate, _lambda, _epsilon, _keep_prob, init_path=None):
        self.graph = tf.Graph()
        self.keep_prob = _keep_prob
        sp_fld_inds = []
        for _i in range(batch_size):
            sp_fld_inds.append([_i, 0])
        sp_eval_fld_inds = []
        for _i in range(eval_size):
            sp_eval_fld_inds.append([_i, 0])
        mbdng_dim = X_feas * (rank + 1) + 1
        h1_dim = 800
        h2_dim = 400
        print mbdng_dim, h1_dim, h2_dim

        if init_path:
            init_model = open(init_path, 'rb')
            var_map = pickle.load(init_model)
            init_model.close()
        else:
            var_map = {}

        with self.graph.as_default():
            self.feed_var_map(var_map, X_dim, rank, mbdng_dim, h1_dim, h2_dim, _min_val, _max_val, _seeds)

            self.fm_w = tf.Variable(var_map['W'])
            self.fm_v = tf.Variable(var_map['V'])
            self.fm_b = tf.Variable(var_map['b'])
            self.h1_w = tf.Variable(var_map['h1_w'])
            self.h1_b = tf.Variable(var_map['h1_b'])
            self.h2_w = tf.Variable(var_map['h2_w'])
            self.h2_b = tf.Variable(var_map['h2_b'])
            self.h3_w = tf.Variable(var_map['h3_w'])
            self.h3_b = tf.Variable(var_map['h3_b'])

            self.v_wt_hldr = tf.placeholder(tf.float32, shape=[batch_size, 13])
            self.c_id_hldr = tf.placeholder(tf.int64, shape=[batch_size, 26])
            self.c_wt_hldr = tf.placeholder(tf.float32, shape=[batch_size, 26])
            self.lbl_hldr = tf.placeholder(tf.float32)

            self.fm_wv = tf.concat(1, [self.fm_w, self.fm_v])
            self.v_fld_w = [tf.slice(self.fm_wv, [_i, 0], [1, rank + 1]) for _i in range(13)]
            self.c_fld_w = [tf.slice(self.fm_wv, [offsets[_i], 0], [cat_sizes[_i], rank + 1]) for _i in
                            range(X_feas - 13)]

            logits = self.forward(batch_size, X_feas, self.v_wt_hldr, self.c_id_hldr, self.c_wt_hldr, sp_fld_inds, True)
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, self.lbl_hldr)) + _lambda * (
                tf.nn.l2_loss(self.fm_w) + tf.nn.l2_loss(self.fm_v) + tf.nn.l2_loss(self.fm_b) + tf.nn.l2_loss(
                    self.h1_w) + tf.nn.l2_loss(self.h1_b) + tf.nn.l2_loss(self.h2_w) + tf.nn.l2_loss(
                    self.h2_b) + tf.nn.l2_loss(self.h3_w) + tf.nn.l2_loss(self.h3_b))

            self.ptmzr = tf.train.AdamOptimizer(learning_rate=_learning_rate, epsilon=_epsilon).minimize(self.loss)
            self.train_preds = tf.sigmoid(logits)
            eval_logits = self.forward(eval_size, X_feas, eval_wts[:, :13], eval_cols[:, 13:] - offsets,
                                       eval_wts[:, 13:], sp_eval_fld_inds)
            self.eval_preds = tf.sigmoid(eval_logits)

    def feed_var_map(self, var_map, X_dim, rank, mbdng_dim, h1_dim, h2_dim, _min_val, _max_val, _seeds):
        if 'W' not in var_map.keys():
            var_map['W'] = tf.random_uniform([X_dim, 1], minval=_min_val, maxval=_max_val, seed=_seeds[0])
        if 'V' not in var_map.keys():
            var_map['V'] = tf.random_uniform([X_dim, rank], minval=_min_val, maxval=_max_val, seed=_seeds[1])
        if 'b' not in var_map.keys():
            var_map['b'] = tf.zeros([1])
        if 'h1_w' not in var_map.keys():
            var_map['h1_w'] = tf.random_uniform([mbdng_dim, h1_dim], minval=_min_val, maxval=_max_val, seed=_seeds[2])
        if 'h1_b' not in var_map.keys():
            var_map['h1_b'] = tf.zeros([h1_dim])
        if 'h2_w' not in var_map.keys():
            var_map['h2_w'] = tf.random_uniform([h1_dim, h2_dim], minval=_min_val, maxval=_max_val, seed=_seeds[3])
        if 'h2_b' not in var_map.keys():
            var_map['h2_b'] = tf.zeros([h2_dim])
        if 'h3_w' not in var_map.keys():
            var_map['h3_w'] = tf.random_uniform([h2_dim, 1], minval=_min_val, maxval=_max_val, seed=_seeds[4])
        if 'h3_b' not in var_map.keys():
            var_map['h3_b'] = tf.zeros([1])

    def forward(self, N, M, v_wts, c_ids, c_wts, sp_inds, drop_out=False):
        v_mbdng = tf.concat(1, [tf.matmul(tf.reshape(v_wts[:, _i], [N, 1]), self.v_fld_w[_i]) for _i in range(13)])
        c_fld_ids = [tf.SparseTensor(sp_inds, c_ids[:, _i], shape=[N, 1]) for _i in range(M - 13)]
        c_fld_wts = [tf.SparseTensor(sp_inds, c_wts[:, _i], shape=[N, 1]) for _i in range(M - 13)]
        c_mbdng = tf.concat(1, [
            tf.nn.embedding_lookup_sparse(self.c_fld_w[_i], c_fld_ids[_i], c_fld_wts[_i], combiner='sum') for _i in
            range(M - 13)])
        b_mbdng = tf.reshape(tf.concat(0, [tf.identity(self.fm_b) for _i in range(N)]), [N, 1])

        z1 = tf.concat(1, [v_mbdng, c_mbdng, b_mbdng])
        if drop_out:
            l2 = tf.matmul(tf.nn.dropout(tf.tanh(z1), keep_prob=self.keep_prob), self.h1_w) + self.h1_b
            l3 = tf.matmul(tf.nn.dropout(tf.tanh(l2), keep_prob=self.keep_prob), self.h2_w) + self.h2_b
            yhat = tf.matmul(tf.nn.dropout(tf.tanh(l3), keep_prob=self.keep_prob), self.h3_w) + self.h3_b
        else:
            l2 = tf.matmul(tf.tanh(z1), self.h1_w) + self.h1_b
            l3 = tf.matmul(tf.tanh(l2), self.h2_w) + self.h2_b
            yhat = tf.matmul(tf.tanh(l3), self.h3_w) + self.h3_b
        return yhat

    def dump(self, model_path):
        var_map = {'W': self.fm_w.eval(), 'V': self.fm_v.eval(), 'b': self.fm_b.eval(), 'h1_w': self.h1_w.eval(),
                   'h1_b': self.h1_b.eval(), 'h2_w': self.h2_w.eval(), 'h2_b': self.h2_b.eval(),
                   'h3_w': self.h3_w.eval(), 'h3_b': self.h3_b.eval()}
        pickle.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at %s' % model_path
