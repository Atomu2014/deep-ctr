import time
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')


def collect(fin, size=1000000):
    buf = []
    for i in range(size):
        try:
            line = next(fin)
            buf.append(line)
        except StopIteration as e:
            break
    return buf


def stat(file_name, max_vals, sets, dump_file):
    fin = open(file_name, 'rb')

    num_lines = 0
    print 'processing', file_name, max_vals, [len(x) for x in sets]
    while True:
        start_time = time.time()
        buf = collect(fin)
        num_lines += len(buf)
        if len(buf) < 1:
            break

        for line in buf:
            fields = line.strip().split('\t')

            vals = [int(max(x, '0')) for x in fields[1:14]]
            max_vals = [max(max_vals[i], vals[i]) for i in range(13)]
            cats = [int(max(x, '0'), 16) for x in fields[14:]]

            for i in range(26):
                if cats[i] in sets[i]:
                    sets[i][cats[i]] += 1
                else:
                    sets[i][cats[i]] = 1
        print num_lines, max_vals, [len(x) for x in sets], time.time() - start_time

    pickle.dump({'max_vals': max_vals, 'sets': sets}, open(dump_file, 'wb'))


def stat2(file_list, dump_file):
	max_vals = [0] * 13
	cats = [{} for i in range(26)]

	for file in file_list:
		stat(file, max_vals, cats, dump_file)


def stat_freq_filter(stat_file):
    sns.color_palette('hls', 26)

    save = pickle.load(open(stat_file, 'rb'))
    sets = save['sets']

    total_weights = [np.sum(sets[i].values()) for i in range(26)]
    total_sizes = [len(sets[i]) for i in range(26)]

    def foo(freq):
        sizes = [np.array(sets[i].values()) for i in range(26)]
        inds = [np.where(sizes[i] > freq)[0] for i in range(26)]
        f_sizes = [len(inds[i]) for i in range(26)]
        f_weights = [np.sum(sizes[i][inds[i]]) for i in range(26)]
        f_s_ratios = [f_sizes[i] * 1.0 / total_sizes[i] for i in range(26)]
        f_w_ratios = [f_weights[i] * 1.0 / total_weights[i] for i in range(26)]
        # print 'dimension %d->%d' % (np.sum(total_sizes), np.sum(f_sizes))
        # print 'field\tt_size\tt_weight\tf_size\tf_weight\tfs_ratio\tfw_ratio'
        # for i in range(26):
        # print '%3d%10d%12d%10d%12d\t%.4f\t%.4f' % (i, total_sizes[i],
        # total_weights[i], f_sizes[i], f_weights[i], f_s_ratios[i],
        # f_w_ratios[i])
        return f_s_ratios, f_w_ratios

    s_ratios = []
    w_ratios = []
    thresholds = [10, 20, 50, 100, 200, 500, 1000]
    for i in thresholds:
        s, w = foo(i)
        s_ratios.append(s)
        w_ratios.append(w)

    s_ratios = np.array(s_ratios)
    w_ratios = np.array(w_ratios)

    for i in range(26):
        plt.plot(thresholds, s_ratios[:, i])
    plt.xlabel('freq threshold')
    plt.ylabel('dim reduce rate')
    plt.title('dim reduction on different fields')
    plt.show()

    for i in range(26):
        plt.plot(thresholds, w_ratios[:, i])
    plt.xlabel('freq threshold')
    plt.ylabel('weight reduce rate')
    plt.title('item weights on different fields')
    plt.show()


if __name__ == '__main__':
    # stat_freq_filter('../data/stat.pickle')
    stat2(['../data/nds.2.5.shuf', '../data/test.nds.2.5.shuf', '../data/test.unif.2.5.shuf'], '../data/2.5.stat.pickle')
