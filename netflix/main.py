import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")
Ks = np.array([1, 2, 3, 4])
seeds = np.array([0, 1, 2, 3, 4])

for K in Ks:
    for seed in seeds:
        mixture, post = common.init(X, K, seed)
        mixture, post, cost = kmeans.run(X, mixture, post)

        title = 'Mixture model with K={}, seed={}, cost={}'.format(K, seed, cost)
        print(title)
        common.plot(X, mixture, post, title=title)
