import cPickle as pickle

import tensorflow as tf


class LR:
    def __init__(self, batch_size, eval_size, X_dim, X_feas, sp_train_inds, sp_eval_inds, eval_cols, eval_wts, _min_val,
                 _max_val, _seeds, _optimizer, _learning_rate, _lambda, _epsilon):
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.sp_id_hldr = tf.placeholder(tf.int64, shape=[batch_size * X_feas])
            self.sp_wt_hldr = tf.placeholder(tf.float32, shape=[batch_size * X_feas])
            sp_ids = tf.SparseTensor(sp_train_inds, self.sp_id_hldr, shape=[batch_size, X_feas])
            sp_wts = tf.SparseTensor(sp_train_inds, self.sp_wt_hldr, shape=[batch_size, X_feas])
            self.lbl_hldr = tf.placeholder(tf.float32)
            sp_eval_ids = tf.SparseTensor(sp_eval_inds, eval_cols, shape=[eval_size, X_feas])
            sp_eval_wts = tf.SparseTensor(sp_eval_inds, eval_wts, shape=[eval_size, X_feas])

            self.W = tf.Variable(tf.random_uniform([X_dim, 1], minval=_min_val, maxval=_max_val, seed=_seeds[0]))
            self.b = tf.Variable(0.0)

            logits = self.regression(sp_ids, sp_wts)
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, self.lbl_hldr)) + _lambda * (
                tf.nn.l2_loss(self.W) + tf.nn.l2_loss(self.b))
            if _optimizer == 'ftrl':
                self.ptmzr = tf.train.FtrlOptimizer(_learning_rate).minimize(self.loss)
            elif _optimizer == 'adam':
                self.ptmzr = tf.train.AdamOptimizer(learning_rate=_learning_rate, epsilon=_epsilon).minimize(self.loss)
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
