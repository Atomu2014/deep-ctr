import linecache
import math
import sys
import time

import numpy
import theano
import theano.tensor as T
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from theano.tensor.shared_randomstreams import RandomStreams

import dl_utils as ut


def lr_via_xgb(eval_metric='auc'):
    train_path = '../data/train.fm.txt'
    test_path = '../data/test.fm.txt'

    dtrain = xgb.DMatrix(train_path)
    dtest = xgb.DMatrix(test_path)

    param = {'silent': 1, 'objective': 'binary:logistic', 'booster': 'gblinear', 'alpha': 0.0001, 'lambda': 1,
             'eval_metric': eval_metric}

    watchlist = [(dtest, 'eval'), (dtrain, 'train')]

    num_round = 100

    bst = xgb.train(param, dtrain, num_round, watchlist)

    bst.dump_model(train_path + '.lr.param')
    bst.save_model(train_path + '.lr.model')


def lr_via_theano():
    class LogisticRegression(object):
        """Multi-class Logistic Regression Class

        The logistic regression is fully described by a weight matrix :math:`W`
        and bias vector :math:`b`. Classification is done by projecting data
        points onto a set of hyperplanes, the distance to which is used to
        determine a class membership probability.
        """

        def __init__(self, input, n_in, n_out):
            self.W = theano.shared(
                    value=numpy.zeros(
                            (n_in, n_out),
                            dtype=theano.config.floatX
                    ),
                    name='W',
                    borrow=True
            )
            self.b = theano.shared(
                    value=numpy.zeros(
                            (n_out,),
                            dtype=theano.config.floatX
                    ),
                    name='b',
                    borrow=True
            )

            self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
            self.y_pred = T.argmax(self.p_y_given_x, axis=1)
            self.params = [self.W, self.b]
            self.input = input

        def negative_log_likelihood(self, y):
            return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

        def errors(self, y):
            if y.ndim != self.y_pred.ndim:
                raise TypeError(
                        'y should have the same shape as self.y_pred',
                        ('y', y.type, 'y_pred', self.y_pred.type)
                )
            if y.dtype.startswith('int'):
                return T.mean(T.neq(self.y_pred, y))
            else:
                raise NotImplementedError()

    srng = RandomStreams(seed=234)
    rng = numpy.random
    rng.seed(1234)
    batch_size = 100  # batch size
    lr = 0.002  # learning rate
    lambda1 = 0.1  # .01                                                        #regularisation rate
    hidden1 = 300  # hidden layer 1
    hidden2 = 100  # hidden layer 2
    acti_type = 'tanh'  # activation type
    epoch = 100  # epochs number
    advertiser = '2997'
    if len(sys.argv) > 1:
        advertiser = sys.argv[1]
    train_file = '../data/train.fm.txt'  # training file
    test_file = '../data/test.fm.txt'  # test file
    fm_model_file = '../data/fm.model.txt'  # fm model file
    # feats = ut.feats_len(train_file)                                           #feature size
    if len(sys.argv) > 2 and advertiser == 'all':
        train_file = train_file + '.5.txt'
    elif len(sys.argv) > 2:
        train_file = train_file + '.10.txt'
    print train_file

    train_size = ut.file_len(train_file)  # training size
    test_size = ut.file_len(test_file)  # test size
    n_batch = train_size / batch_size  # number of batches
    x_drop = 1

    if advertiser == '2997':  #
        lr = 0.001
        x_drop = dropout = 0.5
        hidden1 = 300
        hidden2 = 100
        lambda1 = 0.0
        lambda_fm = 0.1

    name_field = {'weekday': 0, 'hour': 1, 'useragent': 2, 'IP': 3, 'region': 4, 'city': 5, 'adexchange': 6,
                  'domain': 7,
                  'slotid': 8, 'slotwidth': 9, 'slotheight': 10, 'slotvisibility': 11, 'slotformat': 12, 'creative': 13,
                  'advertiser': 14, 'slotprice': 15}

    def log_p(msg, m=""):
        print msg

    # print error
    def print_err(file, msg=''):
        auc, rmse = get_err_bat(file)
        log_p(msg + '\t' + str(auc) + '\t' + str(rmse))

    log_p('ad:' + str(advertiser))
    log_p('batch_size:' + str(batch_size))
    feat_field = {}
    feat_weights = {}
    w_0 = 0
    feat_num = 0
    k = 0
    xdim = 0
    fi = open(fm_model_file, 'r')
    first = True
    for line in fi:
        s = line.strip().split()
        if first:
            first = False
            w_0 = float(s[0])
            feat_num = int(s[1])
            k = int(s[2]) + 1  # w and v
            xdim = 1 + len(name_field) * k
        else:
            feat = int(s[0])
            weights = [float(s[1 + i]) for i in range(k)]
            feat_weights[feat] = weights
            name = s[1 + k][0:s[1 + k].index(':')]
            field = name_field[name]
            feat_field[feat] = field

    def feat_layer_one_index(feat, l):
        return 1 + feat_field[feat] * k + l

    def feats_to_layer_one_array(feats):
        x = numpy.zeros(xdim)
        x[0] = w_0
        for feat in feats:
            x[feat_layer_one_index(feat, 0):feat_layer_one_index(feat, k)] = feat_weights[feat]
        return x

    def get_xy(line):
        s = line.replace(':', ' ').split()
        y = int(s[0])
        feats = [int(s[j]) for j in range(1, len(s), 2)]
        x = feats_to_layer_one_array(feats)
        return x, y

    def get_fxy(line):
        s = line.replace(':', ' ').split()
        y = int(s[0])
        feats = [int(s[j]) for j in range(1, len(s), 2)]
        x = feats_to_layer_one_array(feats)
        return feats, x, y

    def get_batch_data(file, index, size):  # 1,5->1,2,3,4,5
        xarray = []
        yarray = []
        farray = []
        for i in range(index, index + size):
            line = linecache.getline(file, i)
            if line.strip() != '':
                f, x, y = get_fxy(line.strip())
                xarray.append(x)
                yarray.append(y)
                farray.append(f)
        xarray = numpy.array(xarray, dtype=theano.config.floatX)
        yarray = numpy.array(yarray, dtype=numpy.int32)
        return farray, xarray, yarray

    def get_err_bat(file, err_batch=100000):
        y = []
        yp = []
        fi = open(file, 'r')
        flag_start = 0
        xx_bat = []
        flag = False
        while True:
            line = fi.readline()
            if len(line) == 0:
                flag = True
            flag_start += 1
            if flag == False:
                xx, yy = get_xy(line)
                xx_bat.append(numpy.asarray(xx))
            if ((flag_start == err_batch) or (flag == True)):
                pred = predict(xx_bat)
                for p in pred:
                    yp.append(p)
                flag_start = 0
                xx_bat = []
            if flag == False:
                y.append(yy)
            if flag == True:
                break
        fi.close()
        auc = roc_auc_score(y, yp)
        rmse = math.sqrt(mean_squared_error(y, yp))
        return auc, rmse

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    classifier = LogisticRegression(input=x, n_in=177, n_out=1)
    cost = classifier.negative_log_likelihood(y)

    test_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: get_batch_data(test_file, index, batch_size)[1],
                y: get_batch_data(test_file, index, batch_size)[2]
            }
    )

    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)
    updates = [(classifier.W, classifier.W - lr * g_W),
               (classifier.b, classifier.b - lr * g_b)]

    train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: get_batch_data(train_file, index, batch_size)[1],
                y: get_batch_data(train_file, index, batch_size)[2]
            }
    )

    p_1 = 1 / (1 + T.exp(-T.dot(x, classifier.W) - classifier.b))  # Probability that target = 1
    prediction = p_1  # > 0.5                                   # The prediction thresholded
    predict = theano.function(inputs=[x], outputs=prediction)

    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_batch, patience / 2)

    for i in range(epoch):
        start_time = time.time()
        index = 1
        for j in range(n_batch):
            if index > train_size:
                break

            minibatch_avg_cost = train_model(index)
            iter = i * n_batch + index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = [test_model(i) for i in xrange(test_size / batch_size)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        index + 1,
                        n_batch,
                        this_validation_loss * 100.
                    )
                )

            index += batch_size

        train_time = time.time() - start_time
        mins = int(train_time / 60)
        secs = int(train_time % 60)
        print 'training: ' + str(mins) + 'm ' + str(secs) + 's'

        start_time = time.time()
        print_err(train_file, '\t\tTraining Err: \t' + str(i))  # train error
        train_time = time.time() - start_time
        mins = int(train_time / 60)
        secs = int(train_time % 60)
        print 'training error: ' + str(mins) + 'm ' + str(secs) + 's'

        start_time = time.time()
        auc, rmse = get_err_bat(test_file)
        test_time = time.time() - start_time
        mins = int(test_time / 60)
        secs = int(test_time % 60)
        log_p('Test Err:' + str(i) + '\t' + str(auc) + '\t' + str(rmse))
        print 'test error: ' + str(mins) + 'm ' + str(secs) + 's'


if __name__ == '__main__':
    lr_via_theano()
    # lr_via_xgb('auc')
