from tf_util import *


class LR:
    def __init__(self, batch_size, _rch_argv, _init_argv, _ptmzr_argv, _reg_argv, mode, eval_size):
        self.graph = tf.Graph()
        X_dim, X_feas = _rch_argv

        sp_train_inds = build_inds(batch_size, X_feas)
        if mode == 'train':
            sp_eval_inds = build_inds(eval_size, X_feas)

        self.log = 'input dim: %d, features: %d, ' % (X_dim, X_feas)
        with self.graph.as_default():
            self.log = 'input dim: %d, features: %d, ' % (X_dim, X_feas)
            var_map, log = init_var_map(_init_argv, [('W', [X_dim, 1], 'random'),
                                                     ('b', [1], 'zero')])
            self.log += log
            self.W = tf.Variable(var_map['W'])
            self.b = tf.Variable(var_map['b'])

            self.sp_id_hldr = tf.placeholder(tf.int64, shape=[batch_size * X_feas])
            self.sp_wt_hldr = tf.placeholder(tf.float32, shape=[batch_size * X_feas])
            self.lbl_hldr = tf.placeholder(tf.float32)

            sp_ids = tf.SparseTensor(sp_train_inds, self.sp_id_hldr, shape=[batch_size, X_feas])
            sp_wts = tf.SparseTensor(sp_train_inds, self.sp_wt_hldr, shape=[batch_size, X_feas])

            if mode == 'train':
                _lambda = _reg_argv[0]
                logits = self.regression(sp_ids, sp_wts)
                self.train_preds = tf.sigmoid(logits)
                log_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, self.lbl_hldr)
                if _ptmzr_argv[-1] == 'sum':
                    self.loss = tf.reduce_sum(log_loss)
                else:
                    self.loss = tf.reduce_mean(log_loss)
                self.loss += _lambda * (tf.nn.l2_loss(self.W) + tf.nn.l2_loss(self.b))

                self.ptmzr, log = builf_optimizer(_ptmzr_argv, self.loss)
                self.log += '%s, lambda(l2): %g, reduce by: %s' % (log, _lambda, _ptmzr_argv[-1])

                self.eval_id_hldr = tf.placeholder(tf.int64, shape=[eval_size * X_feas])
                self.eval_wt_hldr = tf.placeholder(tf.float32, shape=[eval_size * X_feas])
                sp_eval_ids = tf.SparseTensor(sp_eval_inds, self.eval_id_hldr, shape=[eval_size, X_feas])
                sp_eval_wts = tf.SparseTensor(sp_eval_inds, self.eval_wt_hldr, shape=[eval_size, X_feas])
                eval_logits = self.regression(sp_eval_ids, sp_eval_wts)
                self.eval_preds = tf.sigmoid(eval_logits)
            else:
                logits = self.regression(sp_ids, sp_wts)
                self.test_preds = tf.sigmoid(logits)

    def regression(self, sp_ids, sp_wts):
        yhat = tf.nn.embedding_lookup_sparse(self.W, sp_ids, sp_wts, combiner='sum') + self.b
        return tf.reshape(yhat, [-1, ])

    def dump(self, model_path):
        var_map = {'W': self.W.eval(), 'b': self.b.eval()}
        pickle.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at %s' % model_path
