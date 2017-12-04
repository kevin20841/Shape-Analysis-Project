import numpy as np
from numba.targets.arraymath import np_all

from shapedist.elastic_linear import find_gamma, find_shape_distance
from scipy.interpolate import InterpolatedUnivariateSpline
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
while i < len(data) - 1:
    temp = data[i].split("  ")[1:]
    temp = [float(x) for x in temp]
    temp = np.array(temp)
    temp = (temp - temp.min(0)) / (temp.max(0) - temp.min(0))
    array_of_curves.append(temp)
    i = i + 1
array_of_curves = np.array(array_of_curves)
shape_distance_matrix = np.zeros([len(data)-1, len(data)-1])

i = 90
j = 58
tg_temp, gamma_temp, energy = find_gamma(np.array([t, array_of_curves[i]]),
                  np.array([t, array_of_curves[j]]),
                  10, 12, 6)
q_function = InterpolatedUnivariateSpline(t, array_of_curves[j])
gamma_function = InterpolatedUnivariateSpline(tg_temp, gamma_temp)

temp = q_function(gamma_function(t))
shape_distance_matrix[i][j] = find_shape_distance(t, array_of_curves[i], temp)

print(shape_distance_matrix[i][j], i, j)

plt.plot(t, temp, ".-r")
plt.plot(t, array_of_curves[j], ".-y")
plt.plot(t, array_of_curves[i], ".g")
plt.show()

