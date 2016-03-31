import tensorflow as tf


class FNN:
    def __init__(self, cat_sizes, offsets, batch_size, eval_size, X_dim, X_feas, sp_train_inds, sp_eval_inds, eval_cols,
                 eval_wts, rank, _min_val, _max_val, _seeds, _learning_rate, _alpha, _lambda, _epsilon):
        self.graph = tf.Graph()
        sp_fld_inds = []
        for _i in range(batch_size):
            sp_fld_inds.append([_i, 0])

        with self.graph.as_default():
            tf_fm_w = tf.Variable(tf.random_uniform([X_dim, 1], minval=_min_val, maxval=_max_val, seed=_seeds[0]))
            tf_fm_v = tf.Variable(tf.random_uniform([X_dim, rank], minval=_min_val, maxval=_max_val, seed=_seeds[1]))
            tf_fm_b = tf.Variable(tf.zeros([1]))

            tf_fm_wv = tf.concat(1, [tf_fm_w, tf_fm_v])
            tf_vf_w = [tf.slice(tf_fm_wv, [_i, 0], [1, rank + 1]) for _i in range(13)]
            tf_cf_w = [tf.slice(tf_fm_wv, [offsets[_i], 0], [cat_sizes[_i], rank + 1]) for _i in range(X_feas - 13)]

            self.tf_vf_x = tf.placeholder(tf.float32, shape=[batch_size, 13])
            tf_vf_mbd = tf.concat(1, [tf.matmul(tf.reshape(self.tf_vf_x[:, _i], [batch_size, 1]), tf_vf_w[_i]) for _i in
                                      range(13)])

            self.id_vals = tf.placeholder(tf.int64, shape=[batch_size, 26])
            self.weight_vals = tf.placeholder(tf.float32, shape=[batch_size, 26])
            tmp6 = [tf.SparseTensor(sp_fld_inds, self.id_vals[:, _i], shape=[batch_size, 1]) for _i in
                    range(X_feas - 13)]
            tmp7 = [tf.SparseTensor(sp_fld_inds, self.weight_vals[:, _i], shape=[batch_size, 1]) for _i in
                    range(X_feas - 13)]
            tf_cf_mbd = tf.concat(1, [tf.nn.embedding_lookup_sparse(tf_cf_w[_i], tmp6[_i],
                                                                    tmp7[_i], combiner='sum') for _i in
                                      range(X_feas - 13)])

            tf_fm_b_mbd = tf.reshape(tf.concat(0, [tf.identity(tf_fm_b) for _i in range(batch_size)]), [batch_size, 1])
            self.tf_mbd = tf.concat(1, [tf_vf_mbd, tf_cf_mbd, tf_fm_b_mbd])
