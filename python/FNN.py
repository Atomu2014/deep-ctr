import tensorflow as tf


class FNN:
    def __init__(self, cat_sizes, offsets, batch_size, eval_size, X_dim, X_feas, eval_cols, eval_wts, rank, _min_val,
                 _max_val, _seeds, _learning_rate, _alpha, _lambda):
        self.graph = tf.Graph()
        sp_fld_inds = []
        for _i in range(batch_size):
            sp_fld_inds.append([_i, 0])
        sp_eval_fld_inds = []
        for _i in range(eval_size):
            sp_eval_fld_inds.append([_i, 0])
        mbdng_dim = X_feas * (rank + 1) + 1
        h1_dim = 300
        h2_dim = 100

        with self.graph.as_default():
            fm_w = tf.Variable(tf.random_uniform([X_dim, 1], minval=_min_val, maxval=_max_val, seed=_seeds[0]))
            fm_v = tf.Variable(tf.random_uniform([X_dim, rank], minval=_min_val, maxval=_max_val, seed=_seeds[1]))
            self.fm_b = tf.Variable(tf.zeros([1]))

            self.fm_wv = tf.concat(1, [fm_w, fm_v])
            self.v_fld_w = [tf.slice(self.fm_wv, [_i, 0], [1, rank + 1]) for _i in range(13)]
            self.c_fld_w = [tf.slice(self.fm_wv, [offsets[_i], 0], [cat_sizes[_i], rank + 1]) for _i in
                            range(X_feas - 13)]

            self.v_wt_hldr = tf.placeholder(tf.float32, shape=[batch_size, 13])
            self.c_id_hldr = tf.placeholder(tf.int64, shape=[batch_size, 26])
            self.c_wt_hldr = tf.placeholder(tf.float32, shape=[batch_size, 26])
            self.lbl_hldr = tf.placeholder(tf.float32)

            self.h1_w = tf.Variable(
                tf.random_uniform([mbdng_dim, h1_dim], minval=_min_val, maxval=_max_val, seed=_seeds[2]))
            self.h1_b = tf.Variable(tf.zeros([h1_dim]))

            self.h2_w = tf.Variable(
                tf.random_uniform([h1_dim, h2_dim], minval=_min_val, maxval=_max_val, seed=_seeds[3]))
            self.h2_b = tf.Variable(tf.zeros([h2_dim]))

            self.h3_w = tf.Variable(tf.random_uniform([h2_dim, 1], minval=_min_val, maxval=_max_val, seed=_seeds[4]))
            self.h3_b = tf.Variable(tf.zeros([1]))

            logits = self.forward(batch_size, X_feas, self.v_wt_hldr, self.c_id_hldr, self.c_wt_hldr, sp_fld_inds)
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, self.lbl_hldr)) + _lambda * (
                tf.nn.l2_loss(fm_w) + tf.nn.l2_loss(fm_v) + tf.nn.l2_loss(self.fm_b) + tf.nn.l2_loss(
                    self.h1_w) + tf.nn.l2_loss(self.h1_b) + tf.nn.l2_loss(self.h2_w) + tf.nn.l2_loss(
                    self.h2_b) + tf.nn.l2_loss(self.h3_w) + tf.nn.l2_loss(self.h3_b)) + _alpha * (
                tf.reduce_sum(tf.abs(fm_w)) + tf.reduce_sum(tf.abs(fm_v)) + tf.abs(self.fm_b) + tf.reduce_sum(
                    tf.abs(self.h1_w)) + tf.reduce_sum(tf.abs(self.h1_b)) + tf.reduce_sum(
                    tf.abs(self.h2_w)) + tf.reduce_sum(tf.abs(self.h2_b)) + tf.reduce_sum(tf.abs(self.h3_w)) + tf.abs(
                    self.h3_b))

            self.ptmzr = tf.train.GradientDescentOptimizer(learning_rate=_learning_rate).minimize(self.loss)
            self.train_preds = tf.sigmoid(logits)
            eval_logits = self.forward(eval_size, X_feas, eval_wts[:, :13], eval_cols[:, 13:] - offsets,
                                       eval_wts[:, 13:], sp_eval_fld_inds)
            self.eval_preds = tf.sigmoid(eval_logits)

    def forward(self, N, M, v_wts, c_ids, c_wts, sp_inds):
        v_mbdng = tf.concat(1, [tf.matmul(tf.reshape(v_wts[:, _i], [N, 1]), self.v_fld_w[_i]) for _i in range(13)])

        c_fld_ids = [tf.SparseTensor(sp_inds, c_ids[:, _i], shape=[N, 1]) for _i in range(M - 13)]
        c_fld_wts = [tf.SparseTensor(sp_inds, c_wts[:, _i], shape=[N, 1]) for _i in range(M - 13)]
        c_mbdng = tf.concat(1, [
            tf.nn.embedding_lookup_sparse(self.c_fld_w[_i], c_fld_ids[_i], c_fld_wts[_i], combiner='sum') for _i in
            range(M - 13)])

        # eval_c_fld_ids = [
        #     tf.SparseTensor(sp_eval_fld_inds, eval_cols[:, 13 + _i] - offsets[_i], shape=[eval_size, 1]) for _i in
        #     range(X_feas - 13)]
        # eval_c_fld_wts = [tf.SparseTensor(sp_eval_fld_inds, eval_wts[:, 13 + _i], shape=[eval_size, 1]) for _i in
        #                   range(X_feas - 13)]
        # eval_c_mbdng = tf.concat(1, [
        #     tf.nn.embedding_lookup_sparse(c_fld_w[_i], eval_c_fld_ids[_i], eval_c_fld_wts[_i], combiner='sum') for
        #     _i in range(X_feas - 13)])

        b_mbdng = tf.reshape(tf.concat(0, [tf.identity(self.fm_b) for _i in range(N)]), [N, 1])
        # eval_b_mbdng = tf.reshape(tf.concat(0, [tf.identity(fm_b) for _i in range(eval_size)]), [eval_size, 1])
        z1 = tf.concat(1, [v_mbdng, c_mbdng, b_mbdng])
        # eval_z1 = tf.concat(1, [eval_v_mbdng, eval_c_mbdng, eval_b_mbdng])

        l2 = tf.matmul(tf.tanh(z1), self.h1_w) + self.h1_b
        l3 = tf.matmul(tf.tanh(l2), self.h2_w) + self.h2_b
        yhat = tf.matmul(tf.tanh(l3), self.h3_w) + self.h3_b
        return yhat
