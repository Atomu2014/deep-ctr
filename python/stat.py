import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
sns.color_palette('hls', 26)

save = pickle.load(open('../data/stat.pickle', 'rb'))
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
	# 	print '%3d%10d%12d%10d%12d\t%.4f\t%.4f' % (i, total_sizes[i], total_weights[i], f_sizes[i], f_weights[i], f_s_ratios[i], f_w_ratios[i])
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
