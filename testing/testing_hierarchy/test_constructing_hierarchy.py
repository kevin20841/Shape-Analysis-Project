from shapedist.build_hierarchy_2D import *
import numpy as np
import matplotlib.pyplot as plt
leaf1 = np.loadtxt("leaf1.txt", dtype=np.float64)
leaf2 = np.loadtxt("leaf2.txt", dtype=np.float64)

# print(test_curvature(leaf1))
# print(curvature(leaf1))
tg, gammay, energy = shapedist.elastic_matcher(leaf1, leaf2, 2)

