import random
import time

import numpy as np

# 195841983
max_vals = [65535, 8000, 2330, 746810, 8000, 57199, 5277, 225635, 3565, 14, 310, 25304793, 21836]
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
cat_sizes = np.array(
    [18576837, 29427, 15127, 7295, 19901, 3, 6465, 1310, 61, 11700067, 622921, 219556, 10, 2209, 9779, 71, 4, 963, 14,
     22022124, 4384510, 15960286, 290588, 10829, 95, 34])


def make_data():
    with open('../data/day_0', 'r') as fin:
        with open('../data/day_0_scale', 'w') as fout_scale:
            with open('../data/day_0_scale_concat', 'w') as fout_concat:
                cnt = 0
                sets = [{} for i in range(26)]
                offset = [13 + sum(cat_sizes[:i]) for i in range(26)]
                buf_scale = ''
                buf_concat = ''
                start_time = time.time()
                end_time = 0
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
                    buf_concat += fields[0] + '\t' + '\t'.join(
                        [str(i) + ':' + vals[i] for i in range(13)]) + '\t' + '\t'.join(
                        str(sets[i][cats[i]] + offset[i]) + ':1' for i in range(26)) + '\n'

                    if cnt % 10000 == 0:
                        fout_scale.write(buf_scale)
                        buf_scale = ''
                        fout_concat.write(buf_concat)
                        buf_concat = ''

                    if cnt % 1000000 == 0:
                        end_time = time.time()
                        print cnt, end_time - start_time
                        start_time = end_time

                print cnt
                if cnt % 10000:
                    fout_scale.write(buf_scale)
                    fout_concat.write(buf_concat)


def sample():
    with open('../data/day_0_scale', 'r') as fin:
        with open('../data/day_0_train', 'w') as train_out:
            with open('../data/day_0_test', 'w') as test_out:
                cnt = 0
                buffer = ''
                for line in fin:
                    cnt += 1
                    buffer += line
                    if cnt % 1000 == 0:
                        if random.random() < 0.002:
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


if __name__ == '__main__':
    # sample()
    # pos_neg_ratio('../data/day_0_train')
    # pos_neg_ratio('../data/day_0_test')
    concat('../data/day_0_train', np.where(cat_sizes < 10000)[0])
    concat('../data/day_0_test', np.where(cat_sizes < 10000)[0])
