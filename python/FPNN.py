from tf_util import *


class FPNN:
    def __init__(self, cat_sizes, offsets, batch_size, eval_size, _rch_argv, _init_argv, _ptmzr_argv, _reg_argv):
        sp_fld_inds = []
        for _i in range(batch_size):
            sp_fld_inds.append([_i, 0])
        sp_eval_fld_inds = []
        for _i in range(eval_size):
            sp_eval_fld_inds.append([_i, 0])

        self._lambda, self._keep_prob = _reg_argv
        self.graph = tf.Graph()
        with self.graph.as_default():
            X_dim, X_feas, rank, h1_dim, h2_dim = _rch_argv
            mbdng_dim = X_feas * ((X_feas - 1) / 2 + rank + 1) + 1
            self.log = 'input dim: %d, features: %d, rank: %d, embedding: %d, h1: %d, h2: %d, ' % \
                       (X_dim, X_feas, rank, mbdng_dim, h1_dim, h2_dim)
            var_map, log = init_var_map(_init_argv, [('W', [X_dim, 1], 'random'),
                                                     ('V', [X_dim, rank], 'random'),
                                                     ('b', [1], 'zero'),
                                                     ('h1_w', [mbdng_dim, h1_dim], 'random'),
                                                     ('h1_b', [h1_dim], 'zero'),
                                                     ('h2_w', [h1_dim, h2_dim], 'random'),
                                                     ('h2_b', [h2_dim], 'zero'),
                                                     ('h3_w', [h2_dim, 1], 'random'),
                                                     ('h3_b', [1], 'zero')])
            self.log += log
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
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits, self.lbl_hldr)) + self._lambda * (
                tf.nn.l2_loss(self.fm_w) + tf.nn.l2_loss(self.fm_v) + tf.nn.l2_loss(self.fm_b) + tf.nn.l2_loss(
                    self.h1_w) + tf.nn.l2_loss(self.h1_b) + tf.nn.l2_loss(self.h2_w) + tf.nn.l2_loss(
                    self.h2_b) + tf.nn.l2_loss(self.h3_w) + tf.nn.l2_loss(self.h3_b))

            self.ptmzr, log = builf_optimizer(_ptmzr_argv, self.loss)
            self.log += '%s, lambda(l2): %g, keep_prob(drop_out): %g' % (log, self._lambda, self._keep_prob)

            self.train_preds = tf.sigmoid(logits)

            self.eval_id_hldr = tf.placeholder(tf.int64)
            self.eval_wts_hldr = tf.placeholder(tf.float32)
            eval_logits = self.forward(eval_size, X_feas, self.eval_wts_hldr[:, :13], self.eval_id_hldr[:, 13:] - offsets,
                                       self.eval_wts_hldr[:, 13:], sp_eval_fld_inds)
            self.eval_preds = tf.sigmoid(eval_logits)

    def forward(self, N, M, v_wts, c_ids, c_wts, sp_inds, drop_out=False):
        mbd = [tf.matmul(tf.reshape(v_wts[:, _i], [-1, 1]), self.v_fld_w[_i]) for _i in range(13)]
        c_fld_ids = [tf.SparseTensor(sp_inds, c_ids[:, _i], shape=[-1, 1]) for _i in range(M - 13)]
        c_fld_wts = [tf.SparseTensor(sp_inds, c_wts[:, _i], shape=[-1, 1]) for _i in range(M - 13)]
        mbd.extend([tf.nn.embedding_lookup_sparse(self.c_fld_w[_i], c_fld_ids[_i], c_fld_wts[_i], combiner='sum')
                    for _i in range(M - 13)])
        p_mbd = tf.transpose(tf.concat(0, [tf.concat(1, [tf.matmul(tf.reshape(mbd[_i][_k, :], [1, -1]),
                                                                   tf.reshape(mbd[_j][_k, :], [-1, 1]))
                                                         for _k in range(N)])
                                           for _i in range(len(mbd) - 1) for _j in range(_i + 1, len(mbd))]))
        b_mbd = tf.reshape(tf.concat(0, [tf.identity(self.fm_b) for _i in range(N)]), [-1, 1])
        mbd = tf.concat(1, mbd)
        z1 = tf.concat(1, [mbd, p_mbd, b_mbd])
        if drop_out:
            l2 = tf.matmul(tf.nn.dropout(tf.tanh(z1), keep_prob=self._keep_prob), self.h1_w) + self.h1_b
            l3 = tf.matmul(tf.nn.dropout(tf.tanh(l2), keep_prob=self._keep_prob), self.h2_w) + self.h2_b
            yhat = tf.matmul(tf.nn.dropout(tf.tanh(l3), keep_prob=self._keep_prob), self.h3_w) + self.h3_b
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
