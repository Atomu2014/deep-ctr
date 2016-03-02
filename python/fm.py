from fastFM import sgd
from pyfm import pylibfm
from sklearn.metrics import roc_auc_score

from dl_utils import libsvm_2_sparse


def fm_fastfm(X_train, y_train, X_test, y_test, n_iter=10000000, init_stdev=0.001, rank=2, step_size=0.002,
              path='train'):
    fm = sgd.FMClassification(n_iter=n_iter, init_stdev=init_stdev, l2_reg_w=0, l2_reg_V=0, rank=rank,
                              step_size=step_size)
    fm.fit(X_train, y_train)

    print path
    print n_iter, init_stdev, rank, step_size
    print 'train auc: %.4f\ttest auc: %.4f' % (
        roc_auc_score(y_train, fm.predict_proba(X_train)), roc_auc_score(y_test, fm.predict_proba(X_test)))


def fm_pyfm(X_train, y_train, X_test, y_test, num_factors=10, num_iter=100, learning_rate=0.001, path='train'):
    fm = pylibfm.FM(num_factors=num_factors, num_iter=num_iter, verbose=True, task='classification',
                    initial_learning_rate=learning_rate, learning_rate_schedule='optimal')

    print path
    print num_factors, num_iter, learning_rate

    fm.fit(X_train, y_train)
    from sklearn.metrics import roc_auc_score

    print path
    print num_factors, num_iter, learning_rate

    print 'train auc: %.4f\ttest auc: %.4f' % (
        roc_auc_score(y_train, fm.predict(X_train)), roc_auc_score(y_test, fm.predict(X_test)))


if __name__ == '__main__':
    path = 'train'
    train_path = '../data/day_0_train_concat'
    test_path = '../data/day_0_test_concat'

    with open(train_path, 'r') as fin:
        rows = []
        for line in fin:
            rows.append(line)
        X_train, y_train = libsvm_2_sparse(rows)

    with open(test_path, 'r') as fin:
        rows = []
        for line in fin:
            rows.append(line)
        X_test, y_test = libsvm_2_sparse(rows)

    # num_factors = 10
    # num_iter = 100
    # learning_rate = 0.001
    # fm_pyfm(X_train, y_train, X_test, y_test, num_factors=num_factors, num_iter=num_iter, learning_rate=learning_rate,
    #         path=path)
    # exit(0)

    y_train *= 2
    y_train -= 1
    y_test *= 2
    y_test -= 1

    n_iter = 10000000
    rank = 2
    for init_stdev in [0.000005, 0.000008, 0.00001, 0.00002]:
        for step_size in [0.0011, 0.0012, 0.0013, 0.0014]:
            fm_fastfm(X_train, y_train, X_test, y_test, n_iter=n_iter, init_stdev=init_stdev, rank=rank,
                      step_size=step_size, path=path)
