import shapedist
import numpy as np
import matplotlib.pyplot as plt
leaf1 = np.loadtxt("leaf1.txt", dtype=np.float64)
leaf2 = np.loadtxt("leaf2.txt", dtype=np.float64)

# print(test_curvature(leaf1))
# print(curvature(leaf1))
tg, gammay, energy, original, boolean_mask = shapedist.find_shapedist(leaf1, leaf2, 2, curve_type="SRVF", adaptive=False)

leaf1 = original[1][boolean_mask[1]]
leaf2 = original[2]
new_leaf2 = np.zeros((tg.size, 2))
new_leaf2[:, 0] = np.interp(gammay, tg, leaf2[boolean_mask[1], 0])
new_leaf2[:, 1] = np.interp(gammay, tg, leaf2[boolean_mask[1], 1])
print(leaf1.shape, new_leaf2.shape, tg.size)
print(shapedist.find_shape_distance_SRVF(tg, leaf1, new_leaf2))
