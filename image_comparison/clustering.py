import numpy as np
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.manifold import MDS, SpectralEmbedding
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import image_comparison.curve_processing
from scipy.io import loadmat
from sqlalchemy.sql.functions import concat
from shapedist import elastic_linear
import numpy as np
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
from scipy import stats
from numba import jit

@jit(nopython = True)
def find_nearest(value, array):
    idx = (np.abs(array-value)).argmin()
    if idx < array.size-1:
        idx = idx
    return idx

f = open("../data/shape_array_1.csv", "r")
test = f.read()
f.close()
test = test.strip().split("\n")
for i in range(len(test)):
    test[i] = test[i].strip().split(",")[:-1]
test = np.array(test)
test = test.astype(float)

X = test

dist_matrix = np.exp(- test ** 2 / (2. * test.max() ** 2))
dist_matrix = (dist_matrix + np.transpose(dist_matrix)) / 2

labels = SpectralClustering(n_clusters=10, eigen_solver='arpack', affinity="precomputed").fit_predict(dist_matrix)
print(len(set(labels)))

all_curves = loadmat('Curve_data.mat')
print( all_curves.keys() )
print( all_curves['MPEG7_classes'] )
curves_nonuniform = all_curves['MPEG7_curves_coarsened']
print(curves_nonuniform.size)
print("hi")



x = []
y = []
z = []
temp = []
for i in range(1, len(curves_nonuniform)):
    p = curves_nonuniform[i][0]
    m = MDS(n_components=3, max_iter=300, n_init=4,
            dissimilarity='euclidean')
    s1, theta1 = image_comparison.curve_processing.convert_to_angle_function(p[0], p[1])
    temp = np.array([s1, theta1])
    p = m.fit_transform(temp)
    x.append(p[0])
    y.append(p[1])
print(len(p))

plt.scatter(x, y, c='r', marker='o', s=2)

plt.show()


