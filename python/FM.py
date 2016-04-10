import cPickle as pickle

import tensorflow as tf


class FM:
    def __init__(self, batch_size, eval_size, X_dim, X_feas, sp_train_inds, sp_eval_inds, eval_cols, eval_wts, rank,
                 _min_val, _max_val, _seeds, _learning_rate, _lambda, _epsilon):
        self.graph = tf.Graph()
        eval_wts2 = eval_wts ** 2

        with self.graph.as_default():
            self.sp_id_hldr = tf.placeholder(tf.int64, shape=[batch_size * X_feas])
            self.sp_wt_hldr = tf.placeholder(tf.float32, shape=[batch_size * X_feas])
            sp_ids = tf.SparseTensor(sp_train_inds, self.sp_id_hldr, shape=[batch_size, X_feas])
            sp_wts = tf.SparseTensor(sp_train_inds, self.sp_wt_hldr, shape=[batch_size, X_feas])
            self.lbl_hldr = tf.placeholder(tf.float32)
            self.sp_wt2_hldr = tf.placeholder(tf.float32, shape=[batch_size * X_feas])
            sp_wts2 = tf.SparseTensor(sp_train_inds, self.sp_wt2_hldr, shape=[batch_size, X_feas])
            sp_eval_ids = tf.SparseTensor(sp_eval_inds, eval_cols, shape=[eval_size, X_feas])
            sp_eval_wts = tf.SparseTensor(sp_eval_inds, eval_wts, shape=[eval_size, X_feas])
            sp_eval_wts2 = tf.SparseTensor(sp_eval_inds, eval_wts2, shape=[eval_size, X_feas])

            self.W = tf.Variable(tf.random_uniform([X_dim, 1], minval=_min_val, maxval=_max_val, seed=_seeds[0]))
            self.V = tf.Variable(tf.random_uniform([X_dim, rank], minval=_min_val, maxval=_max_val, seed=_seeds[1]))
            self.b = tf.Variable(0.0)

            logits = self.factorization(sp_ids, sp_wts, sp_wts2)
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, self.lbl_hldr)) + _lambda * (
                tf.nn.l2_loss(self.W) + tf.nn.l2_loss(self.V) + tf.nn.l2_loss(self.b))
            # self.ptmzr = tf.train.AdamOptimizer(learning_rate=_learning_rate, beta1=_beta1, beta2=_beta2, epsilon=_epsilon).minimize(self.loss)
            self.ptmzr = tf.train.FtrlOptimizer(_learning_rate).minimize(self.loss)
            self.train_preds = tf.sigmoid(logits)
            eval_logits = self.factorization(sp_eval_ids, sp_eval_wts, sp_eval_wts2)
            self.eval_preds = tf.sigmoid(eval_logits)

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
