import numpy as np
from numba.targets.arraymath import np_all
from joblib import Parallel, delayed
from shapedist_old.elastic_linear import find_gamma, find_shape_distance
from scipy.interpolate import InterpolatedUnivariateSpline
np.seterr(divide='ignore', invalid='ignore')
def task(p, q, t):
    temp = p
    gradient1 = np.gradient(temp, t)

    srvf1 = gradient1 / np.sqrt(np.abs(gradient1))

    srvf1[np.isnan(srvf1)] = 0
    srvf1 = (srvf1 - srvf1.min()) / (srvf1.max() - srvf1.min())

    temp = q
    gradient2 = np.gradient(temp, t)

    srvf2 = gradient2 / np.sqrt(np.abs(gradient2))
    srvf2[np.isnan(srvf2)] = 0
    srvf2 = (srvf2 - srvf2.min()) / (srvf2.max() - srvf2.min())
    tg_temp, gamma_temp, energy = find_gamma(np.array([t, srvf1]),
                                             np.array([t, srvf2]),
                                             4, 12, 6)
    q_function = InterpolatedUnivariateSpline(t, srvf2)
    gamma_function = InterpolatedUnivariateSpline(tg_temp, gamma_temp)

    temp = q_function(gamma_function(t))
    return find_shape_distance(t, srvf1, temp)

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
shape_distance_matrix = np.zeros([len(data)-1, len(data)-1])
energy_matrix = np.zeros([len(data)-1, len(data)-1])
i = 0
#test = [[task(x, y, t) for x in array_of_curves[:2]] for y in array_of_curves[:2]]
test = Parallel(n_jobs=8)(delayed(task)(x, y, t) for x in array_of_curves for y in array_of_curves)


f = open("../data/material_science_distance_matrix.txt", "w")
i = 0
temp =int(len(test)**0.5)
while i < temp:
    j = 0
    while j < temp:
        f.write(str(test[int(i * temp + j)]) + "   ")
        j = j + 1
    f.write("\n")
    i = i + 1
f.close()

