from tf_util import *


class FM:
    def __init__(self, batch_size, _rch_argv, _init_argv, _ptmzr_argv, _reg_argv, mode, eval_size, eval_cols, eval_wts):
        self.graph = tf.Graph()
        X_dim, X_feas, rank = _rch_argv

        eval_wts2 = eval_wts ** 2
        sp_train_inds = build_inds(batch_size, X_feas)
        if mode == 'train':
            sp_eval_inds = build_inds(eval_size, X_feas)

        with self.graph.as_default():
            _lambda = _reg_argv[0]
            self.log = 'input dim: %d, features: %d, rank: %d, ' % (X_dim, X_feas, rank)
            var_map, log = init_var_map(_init_argv, [('W', [X_dim, 1], 'random'),
                                                     ('V', [X_dim, rank], 'random'),
                                                     ('b', [1], 'zero')])
            self.log += log
            self.W = tf.Variable(var_map['W'])
            self.V = tf.Variable(var_map['V'])
            self.b = tf.Variable(var_map['b'])

            self.sp_id_hldr = tf.placeholder(tf.int64, shape=[batch_size * X_feas])
            self.sp_wt_hldr = tf.placeholder(tf.float32, shape=[batch_size * X_feas])
            self.lbl_hldr = tf.placeholder(tf.float32)
            self.sp_wt2_hldr = tf.placeholder(tf.float32, shape=[batch_size * X_feas])

            sp_ids = tf.SparseTensor(sp_train_inds, self.sp_id_hldr, shape=[batch_size, X_feas])
            sp_wts = tf.SparseTensor(sp_train_inds, self.sp_wt_hldr, shape=[batch_size, X_feas])
            sp_wts2 = tf.SparseTensor(sp_train_inds, self.sp_wt2_hldr, shape=[batch_size, X_feas])

            if mode == 'train':
                sp_eval_ids = tf.SparseTensor(sp_eval_inds, eval_cols, shape=[eval_size, X_feas])
                sp_eval_wts = tf.SparseTensor(sp_eval_inds, eval_wts, shape=[eval_size, X_feas])
                sp_eval_wts2 = tf.SparseTensor(sp_eval_inds, eval_wts2, shape=[eval_size, X_feas])

                logits = self.factorization(sp_ids, sp_wts, sp_wts2)
                self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, self.lbl_hldr)) + _lambda * (
                    tf.nn.l2_loss(self.W) + tf.nn.l2_loss(self.V) + tf.nn.l2_loss(self.b))
                self.ptmzr, log = builf_optimizer(_ptmzr_argv, self.loss)
                self.log += '%s, lambda(l2): %g' % (log, _lambda)
                self.train_preds = tf.sigmoid(logits)
                eval_logits = self.factorization(sp_eval_ids, sp_eval_wts, sp_eval_wts2)
                self.eval_preds = tf.sigmoid(eval_logits)
            else:
                logits = self.factorization(sp_ids, sp_wts, sp_wts2)
                self.test_preds = tf.sigmoid(logits)

    def factorization(self, sp_ids, sp_weights, sp_weights2):
        yhat = tf.nn.embedding_lookup_sparse(self.W, sp_ids, sp_weights, combiner='sum') + self.b
        _Vx = tf.nn.embedding_lookup_sparse(self.V, sp_ids, sp_weights, combiner='sum')
        _V2x2 = tf.nn.embedding_lookup_sparse(tf.square(self.V), sp_ids, sp_weights2, combiner='sum')
        yhat += 0.5 * tf.reshape(tf.reduce_sum(tf.matmul(_Vx, _Vx, transpose_b=True), 1) - tf.reduce_sum(_V2x2, 1),
                                 shape=[-1, 1])
        return yhat

    def dump(self, model_path):
        var_map = {'W': self.W.eval(), 'V': self.V.eval(), 'b': self.b.eval()}
        pickle.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at %s' % model_path
