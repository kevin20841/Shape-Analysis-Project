import numpy as np
from numba.targets.arraymath import np_all

from shapedist.elastic_linear import find_gamma, find_shape_distance
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

f = open("../data/CURVES_FeGaPd.txt", "r")

data = f.read().split("\n")
f.close()

t = data[0].split("  ")[1:]

t = [float(x) for x in t]
t = np.array(t)
t = (t - t[0]) / (t[t.size-1] - t[0])
array_of_curves = []
i = 1
while i < len(data):
    temp = data[i].split("  ")[1:]
    temp = [float(x) for x in temp]
    temp = np.array(temp)
    temp = (temp - temp.min())
    temp = temp / (temp.max())
    array_of_curves.append(temp)
    i = i + 1
array_of_curves = np.array(array_of_curves)
temp = array_of_curves[0]
gradient1 = np.gradient(temp, t)
gradient1[gradient1 == 0] = 1
srvf1 = gradient1 / np.sqrt(np.abs(gradient1))
srvf1 = (srvf1 - srvf1.min()) / (srvf1.max() - srvf1.min())
tg_temp, gamma_temp, energy = find_gamma(np.array([t, srvf1]),
                          np.array([t, srvf1]),
                          4, 12, 6)
q_function = InterpolatedUnivariateSpline(t, srvf1)
gamma_function = InterpolatedUnivariateSpline(tg_temp, gamma_temp)

temp = q_function(gamma_function(t))
print(find_shape_distance(t, srvf1, temp))
print(len(array_of_curves))

plt.plot(t, srvf1)
plt.plot(tg_temp, gamma_temp)
plt.show()