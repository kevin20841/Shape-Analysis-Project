import sys
sys.path.append("../")
import shapedist
import numpy as np
from scipy.io import loadmat
import time
from tqdm import trange
all_curves = loadmat('../data/Curve_data.mat')

curves_1024 = all_curves['MPEG7_curves_1024']
print(all_curves['MPEG7_classes'])
curves = np.empty((100, 1024, 2))

for i in range(100):
    curves[i] = curves_1024[i][0].T

matrix = np.empty((100, 100))
start = time.time()
for i in trange(100):
    for j in range(100):
        matrix[i][j], temp = shapedist.kd_tree.shape_dist(curves[i], curves[j])
end = time.time()

print("Total Time:", end - start)
np.savetxt("matrix_kd.out", matrix)

# TODO interpolate into arclen
