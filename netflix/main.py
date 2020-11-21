import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")
Ks = np.array([1, 2, 3, 4])
seeds = np.array([0, 1, 2, 3, 4])

# for K in Ks:
#     for seed in seeds:
#         mixture, post = common.init(X, K, seed)
#         mixture, post, cost = kmeans.run(X, mixture, post)
#
#         title = 'Mixture model with K={}, seed={}, cost={}'.format(K, seed, cost)
#         print(title)
#         common.plot(X, mixture, post, title=title)


# for K in Ks:
#     for seed in seeds:
#         mixture, post = common.init(X, K, seed)
#         mixtureG, postG, LL = naive_em.run(X, mixture, post)
#         mixtureK, postK, cost = kmeans.run(X, mixture, post)
#
#         titleG = 'EM Mixture model with K={}, seed={}, LL={}'.format(K, seed, LL)
#         titleK = 'K-Means model with K={}, seed={}, cost={}'.format(K, seed, cost)
#         print(titleG)
#         print(titleK)
#         common.multi_plot(X, mixtureG, mixtureK, postG, postK, titleG, titleK)

# seed = seeds[0]
# for K in Ks:
#     mixture, post = common.init(X, K, seed)
#     mixtureG, postG, LL = naive_em.run(X, mixture, post)
#     bic = common.bic(X, mixture, LL)
#     title = 'EM Mixture model with K={}, seed={},\n LL={}, BIC={}'.format(K, seed, LL, bic)
#     print(title)
#     common.plot(X, mixture, post, title=title)

X = np.loadtxt('netflix_incomplete.txt')
X_gold = np.loadtxt('netflix_complete.txt')
K = 12
best_seed = 0
best_LL = np.NINF
seed = seeds[0]

for seed in seeds:
    mixture, post = common.init(X, K, seed)
    mixture, post, LL = em.run(X, mixture, post)
    complete_matrix = em.fill_matrix(X, mixture)
    print('K={}, seed={}, log-likelihood={}'.format(K, seed, LL))
    if LL > best_LL:
        best_LL = LL
        best_seed = seed
    print('seed:{}, best_seed:{}'.format(seed, best_seed))

print('best seed:', best_seed)

mixture, post = common.init(X, K, best_seed)
mixture, post, LL = em.run(X, mixture, post)
print('input:\n', X)
X_pred = em.fill_matrix(X, mixture)
print('\nK={}, seed={}, log-likelihood={}\n'.format(K, seed, LL))
print('prediction:\n', X_pred)
RMSE = common.rmse(X_gold, X_pred)
print('\nRMSE between X_gold and X_pred =', RMSE)
