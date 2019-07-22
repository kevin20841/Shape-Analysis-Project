import numpy as np
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from sklearn import manifold
import time
import matplotlib.pyplot as plt
from scipy.io import loadmat
from numba import njit
import shapedist

all_curves = loadmat('../data/Curve_data.mat')

curves_1024 = all_curves['MPEG7_curves_1024']
print(all_curves['MPEG7_classes'])
curves = np.empty((100, 1024, 2))

for i in range(100):
    curves[i] = curves_1024[i][0].T
p = curves[93]
q = curves[95]
# p = curves[90]
# q = curves[95]
# p = curves[21]
# q = curves[71]
# p = curves[45]
# q = curves[55]

energy, p_new, q_new, tg, gammay = shapedist.find_shapedist(p, q, 'ud')
plt.plot(tg, gammay, ".b")
plt.figure()
dist1, ind = shapedist.kd_tree.shape_dist(p, q)

print(dist1, energy)


#
# plt.figure()
#
# energy, q_new, p_new, tg, gammay = shapedist.find_shapedist(q, p, 'ud')
# print(p_new.shape)
# dist2, ind = shapedist.kd_tree. shape_dist(q, p)
#
# plt.plot(np.arange(0, ind.shape[0])/(ind.shape[0]-1), ind/ind.shape[0], ".")
# print(dist2, energy)
# plt.plot(tg, gammay, ".")
plt.figure()
plt.plot(p[:, 0], p[:, 1])
plt.plot(q[:, 0], q[:, 1])
plt.show()
