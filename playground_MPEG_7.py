import sys
sys.path.append("./")
import matplotlib.pyplot as plt
import shapedist
import numpy as np
from scipy.io import loadmat
import scipy.interpolate

all_curves = loadmat('./data/Curve_data.mat')

curves_1024 = all_curves['MPEG7_curves_1024']
print(all_curves['MPEG7_classes'])
curves = np.empty((100, 1024, 2))

for i in range(100):
    curves[i] = curves_1024[i][0].T

# for i in range(10, 20):
#     plt.figure()
#     plt.plot(curves[i][:, 0], curves[i][:, 1])

b1 = curves[10]
b2 = curves[12]
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
energy2, p, q, tg, gammay = shapedist.find_shapedist(b1, b2, 'd')
print("Nonuniform:", energy2)
plt.plot(tg, gammay, ".-g")
# plt.figure()
# plt.ylim(-0.2, 0.2)
# plt.xlim(-0.2, 0.2)
# plt.plot(pu[:, 0], pu[:, 1], ".-")
# plt.plot(qu[:, 0], qu[:, 1], ".-")
plt.figure()
plt.ylim(-0.2, 0.2)
plt.xlim(-0.2, 0.2)
plt.plot(p[:, 0], p[:, 1], ".-")
plt.plot(q[:, 0], q[:, 1], ".-")
plt.show()
