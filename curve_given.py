import examples as ex
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

n = 128
t = np.linspace(0.,1.,n)

g = ex.curve_example('bumps', t)[0]
print(len(g[0]))

plt.plot(t,g[0])
plt.show()