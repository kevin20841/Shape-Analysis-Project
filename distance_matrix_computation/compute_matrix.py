import sys
sys.path.append("../")
from joblib import Parallel, delayed, dump, load
import os
import shapedist
import numpy as np
folder = "./jm"
try:
    os.mkdir(folder)
except FileExistsError:
    pass
dfm = os.path.join(folder, 'data_memmap')
from scipy.io import loadmat

all_curves = loadmat('../data/Curve_data.mat')

curves_1024 = all_curves['MPEG7_curves_1024']
print(all_curves['MPEG7_classes'])
curves = np.empty((100, 1024, 2))

for i in range(100):
    curves[i] = curves_1024[i][0].T
dump(curves, dfm)
curves = load(dfm, mmap_mode='r')
shapedist.find_shapedist(curves[0], curves[3], 'u')

matrix = Parallel(n_jobs=-1, verbose=5, max_nbytes=1e5)(delayed(shapedist.find_shapedist)(curves[i//100], curves[i-100 * (i//100)], 'u')for i in range(10000))
matrix = np.array(matrix)
print(matrix)
np.savetxt("matrix_elastic.out", matrix)
