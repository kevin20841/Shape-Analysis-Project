import sys
sys.path.append("./")
import matplotlib.pyplot as plt
import shapedist
import numpy as np
from scipy.io import loadmat
import scipy.interpolate
import time
all_curves = loadmat('../data/Curve_data.mat')

keys = ['MPEG7_curves_128', 'MPEG7_curves_256', 'MPEG7_curves_512', 'MPEG7_curves_1024']

curves_1024 = all_curves['MPEG7_curves_1024']

shape_rep = [shapedist.coords, shapedist.tangent, shapedist.curvature, shapedist.srvf]
shape_rep = [shapedist.coords]
plt.plot(all_curves['MPEG7_curves_128'][10][0].T)
for sr in shape_rep:
    print("Testing bat", sr.__name__.lower())
    for key in keys:
        print(key)
        p = all_curves[key][10][0].T
        q = all_curves[key][11][0].T
        m = np.inf
        for i in range(2):
            t = np.linspace(0., 1., p.shape[0])
            start = time.time()
            sdist = shapedist.closed_curve_shapedist(p, q, dr="um", shape_rep = sr)
            end = time.time()
            if end - start < m:
                m = end - start
        print(str(sdist) + ",", m)
    print()