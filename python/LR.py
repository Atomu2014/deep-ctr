from tf_util import *


class LR:
    def __init__(self, batch_size, eval_size, sp_train_inds, sp_eval_inds, eval_cols, eval_wts, _rch_argv, _init_argv,
                 _ptmzr_argv, _reg_argv):
        self.graph = tf.Graph()

        X_dim, X_feas = _rch_argv
        _lambda = _reg_argv[0]
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
            sp_eval_ids = tf.SparseTensor(sp_eval_inds, eval_cols, shape=[eval_size, X_feas])
            sp_eval_wts = tf.SparseTensor(sp_eval_inds, eval_wts, shape=[eval_size, X_feas])

            logits = self.regression(sp_ids, sp_wts)
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, self.lbl_hldr)) + _lambda * (
                tf.nn.l2_loss(self.W) + tf.nn.l2_loss(self.b))

            self.ptmzr, log = builf_optimizer(_ptmzr_argv, self.loss)
            self.log += '%s, lambda(l2): %g' % (log, _lambda)

            self.train_preds = tf.sigmoid(logits)
            eval_logits = self.regression(sp_eval_ids, sp_eval_wts)
            self.eval_preds = tf.sigmoid(eval_logits)

    def regression(self, sp_ids, sp_wts):
        yhat = tf.nn.embedding_lookup_sparse(self.W, sp_ids, sp_wts, combiner='sum') + self.b
        return yhat

    def dump(self, model_path):
        var_map = {'W': self.W.eval(), 'b': self.b.eval()}
        pickle.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at %s' % model_path
