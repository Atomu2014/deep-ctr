import cPickle as pickle
import random
import time

import numpy as np

# 195841983
max_vals = [65535, 8000, 2330, 746810, 8000, 57199, 5277, 225635, 3565, 14, 310, 25304793, 21836]
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
cat_sizes = np.array(
    [18576837, 29427, 15127, 7295, 19901, 3, 6465, 1310, 61, 11700067, 622921, 219556, 10, 2209, 9779, 71, 4, 963, 14,
     22022124, 4384510, 15960286, 290588, 10829, 95, 34])


def stat(file_name, sets, long_tails, max_vals):
    cnt = 0
    with open(file_name, 'r') as fin:
        start_time = time.time()
        for line in fin:
            fields = line.strip().split('\t')

            vals = [int(max(x, '0')) for x in fields[1:14]]
            max_vals = [max(max_vals[i], vals[i]) for i in range(13)]
            cats = [int(max(x, '0'), 16) for x in fields[14:]]

            for i in range(26):
                if cats[i] in long_tails[i]:
                    long_tails[i].remove(cats[i])
                    sets[i][cats[i]] = 2
                elif cats[i] in sets[i]:
                    sets[i][cats[i]] += 1
                else:
                    long_tails[i].add(cats[i])

            cnt += 1
            if cnt % 10000000 == 0:
                end_time = time.time()
                print cnt, end_time - start_time
                start_time = end_time
                print max_vals
                print [len(x) for x in sets]
                print [len(x) for x in long_tails]
    print 'finish', file_name, max_vals, [len(x) for x in sets], [len(x) for x in long_tails]


def make_data(file_name):
    with open(file_name, 'r') as fin:
        with open(file_name + '.one-hot', 'w') as fout_scale:
            cnt = 0
            sets = [{} for i in range(26)]
            buf_scale = ''
            start_time = time.time()
            for line in fin:
                fields = line.strip().split('\t')

                vals = [max(x, '0') for x in fields[1:14]]
                # vals = [int(x) for x in vals]
                cats = [int(max(x, '0'), 16) for x in fields[14:]]

                for i in range(26):
                    if cats[i] not in sets[i]:
                        sets[i][cats[i]] = len(sets[i])

                cnt += 1

                buf_scale += fields[0] + '\t' + '\t'.join(vals) + '\t' + '\t'.join(
                    str(sets[i][cats[i]]) for i in range(26)) + '\n'

                if cnt % 10000 == 0:
                    fout_scale.write(buf_scale)
                    buf_scale = ''

                if cnt % 1000000 == 0:
                    end_time = time.time()
                    print cnt, end_time - start_time
                    start_time = end_time

            print cnt
            if cnt % 10000:
                fout_scale.write(buf_scale)


def sample(train_path, test_path, alpha=0.02):
    with open('../data/day_0_scale', 'r') as fin:
        with open(train_path, 'w') as train_out:
            with open(test_path, 'w') as test_out:
                cnt = 0
                buffer = ''
                for line in fin:
                    cnt += 1
                    buffer += line
                    if cnt % 1000 == 0:
                        if random.random() < alpha:
                            if random.random() < 0.3:
                                test_out.write(buffer)
                            else:
                                train_out.write(buffer)
                        buffer = ''
                    if cnt % 1000000 == 0:
                        print cnt


def pos_neg_ratio(file_name):
    with open(file_name, 'r') as fin:
        pos = 0
        cnt = 0
        for line in fin:
            if line.split()[0] == '1':
                pos += 1
            cnt += 1
    print pos, cnt, pos * 1.0 / cnt


def concat(file_name, mask):
    print mask, np.sum(cat_sizes[mask])
    offsets = [13 + sum(cat_sizes[mask[:i]]) for i in range(len(mask))]

    with open(file_name, 'r') as fin:
        with open(file_name + '_concat', 'w') as fout:
            buffer = ''
            cnt = 0
            for line in fin:
                fields = line.split('\t')
                vals = [float(fields[i]) / max_vals[i - 1] for i in range(1, 14)]
                cats = fields[14:]
                buffer += fields[0] + ' ' + ' '.join(
                    [str(i) + ':' + str(vals[i]) for i in range(13)]) + ' ' + ' '.join(
                    str(int(cats[mask[i]]) + offsets[i]) + ':1' for i in range(len(mask))) + '\n'
                cnt += 1
                if cnt % 10000 == 0:
                    fout.write(buffer)
                    buffer = ''
                    print cnt

            print cnt
            if cnt % 10000:
                fout.write(buffer)


def nds(fin_path, fout_path, w):
    cnt = 0

    def nds_sample(lines):
        _buf_str = ''
        _buf_size = 0
        for line in lines:
            if int(line.strip().split()[0]) == 0:
                if random.random() < w:
                    _buf_size += 1
                    _buf_str += line
            else:
                _buf_size += 1
                _buf_str += line
        return _buf_str, _buf_size

    with open(fin_path, 'rb') as fin:
        with open(fout_path, 'ab') as fout:
            start_time = time.time()
            buf = []
            buf_str = ''
            buf_size = 0
            for line in fin:
                buf.append(line)
                cnt += 1

                if cnt % 10000 == 0:
                    s, n = nds_sample(buf)
                    buf_str += s
                    buf_size += n
                    buf = []
                    if buf_size > 10000:
                        fout.write(buf_str)
                        buf_str = ''
                        buf_size = 0

                    if cnt % 10000000 == 0:
                        end_time = time.time()
                        print cnt, end_time - start_time
                        start_time = end_time

            if len(buf) > 0:
                s, n = nds_sample(buf)
                buf_str += s
                buf_size += n
            if buf_size > 0:
                fout.write(buf_str)


def seg_file(fin_path):
    def output(buf, fout_name):
        with open(fout_name, 'wb') as fout:
            buf_str = ''
            buf_size = 0
            for line in buf:
                buf_str += line
                buf_size += 1

                if buf_size % 10000 == 0:
                    fout.write(buf_str)
                    buf_str = ''
                    buf_size = 0
            if buf_size:
                fout.write(buf_str)

    with open(fin_path, 'rb') as fin:
        cnt = 0
        buf = []
        for line in fin:
            cnt += 1
            buf.append(line)

            if cnt % 4000000 == 0:
                np.random.shuffle(buf)
                output(buf, fin_path + '.%d' % (cnt / 4000000))
                buf = []
        if len(buf):
            output(buf, fin_path + '.%d' % (cnt / 4000000 + 1))

if __name__ == '__main__':
    # w = 0.1
    # for i in range(7, 14):
    #     print 'processing day_' + str(i)
    #     nds('../data/day_' + str(i), '../data/nds.10', w)
    # nds('../data/day_14', '../data/test.nds.10', w)
    seg_file('../data/nds.10')
    seg_file('../data/test.nds.10')