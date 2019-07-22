import numpy as np
from sklearn.neighbors import KDTree
import examples as ex
import matplotlib.pyplot as plt
import time
n = 2048
t = np.linspace(0., 1., n)
t2 = np.linspace(0., 1., n)
q = ex.curve_example("circle", t)[0].T
p = ex.curve_example("ellipse", t2)[0].T


tree = KDTree(q, leaf_size=2)
start = time.time()
dist, ind = tree.query(p, k=1)
end = time.time()

print(end - start)
ind = np.array(ind)


plt.plot(np.arange(0, n), ind[:, 0], ".-")
plt.show()
