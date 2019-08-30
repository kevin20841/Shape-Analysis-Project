import sys
sys.path.append("./")
import matplotlib.pyplot as plt
import shapedist
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
import scipy.interpolate

# all_curves = loadmat('./data/Curve_data.mat')
#
# curves_1024 = all_curves['MPEG7_curves_1024']
# print(all_curves.keys())
# # curves = np.empty((100, 256, 2))
# #
# # for i in range(100):
# #     curves[i] = curves_1024[i][0].T
#
# p = curves_1024[13][0].T
# q = curves_1024[17][0].T
#
# s = 0- np.argmax(q[:, 0])
# q = np.roll(q, s, axis=0)
# R = np.array([[0.877, -0.479],[0.479, 0.877]])
# R = np.array([[0.877, -0.479],[0.479, 0.877]])
# q = np.zeros(p.shape)
# for i in range(256):
#     q[i] = R @ p[i]

curves = np.load("./data/marrow_cell_curves_full.npy", allow_pickle=True)
p = curves[442]

print(p.shape)
q = curves[43]

t1 = np.linspace(0., 1., p.shape[0])
t2 = np.linspace(0., 1., q.shape[0])
dist, temp, temp, tg, gamma = shapedist.find_shapedist(p, q, t1=t1, t2=t2, dr="cd", tol=2e-4, shape_rep=shapedist.srvf)

print(dist)
print(tg.shape)
plt.plot(tg, gamma, "-")
plt.figure()
plt.plot(p[:, 0], p[:, 1], "r")
plt.plot(q[:, 0], q[:, 1], "g")
plt.show()
# for i in range(519):
#     q = curves[i]
#     dist, temp, temp, tg, gamma = shapedist.find_shapedist(p, q, t1=t, t2=t, dr="d", shape_rep=shapedist.srvf)
#     print(dist)
#     if 0 in np.gradient(gamma, tg):
#         print(i)
#         break
# plt.plot(p[:, 0], p[:, 1])
# plt.plot(q[:, 0], q[:, 1])

# for i in tqdm(range(100)):
#     p = curves[i]
#     for j in range(100):
#         q = curves[j]
#         [t, pn, qn], mask = shapedist.build_hierarchy.get_adaptive_mask(p, q, shapedist.arclen_fct_values(p), shapedist.arclen_fct_values(q))
#         s = pn[mask[0]].shape[0]
#         s2 = pn[mask[1]].shape[0]
#     print(i, j, s)
# for i in range(10, 20):
#     plt.figure()
#     plt.plot(curves[i][:, 0], curves[i][:, 1])

# b1 = curves[10]
# b2 = curves[12]
# t1 = shapedist.arclen_fct_values(b1)
# t2 = shapedist.arclen_fct_values(b2)
#
# b1f1 = scipy.interpolate.CubicSpline(t1, b1[:, 0])
# b1f2 = scipy.interpolate.CubicSpline(t1, b1[:, 1])
#
# b2f1 = scipy.interpolate.CubicSpline(t2, b2[:, 0])
# b2f2 = scipy.interpolate.CubicSpline(t2, b2[:, 1])
# t = np.linspace(0., 1., 1039)
# b1 = np.zeros((1039, 2))
# b2 = np.zeros((1039, 2))
# b1[:, 0] = b1f1(t)
# b1[:, 1] = b1f2(t)
# b2[:, 0] = b2f1(t)
# b2[:, 1] = b2f2(t)

# energy1, pu, qu, tg, gammay = shapedist.find_shapedist(b1, b2, 'd')
# print("Nonuniform:", energy1)
# plt.plot(tg, gammay, "-r")
# energy2, p, q, tg, gammay = shapedist.find_shapedist(b1, b2, 'd')
# print("Nonuniform:", energy2)
# plt.plot(tg, gammay, ".-g")
# # plt.figure()
# # plt.ylim(-0.2, 0.2)
# # plt.xlim(-0.2, 0.2)
# # plt.plot(pu[:, 0], pu[:, 1], ".-")
# # plt.plot(qu[:, 0], qu[:, 1], ".-")
# plt.figure()
# plt.ylim(-0.2, 0.2)
# plt.xlim(-0.2, 0.2)
# plt.plot(p[:, 0], p[:, 1], ".-")
# plt.plot(q[:, 0], q[:, 1], ".-")
# plt.show()
