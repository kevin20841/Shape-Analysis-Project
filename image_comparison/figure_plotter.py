import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from shapedist.elastic_linear import find_gamma
from testing import examples as ex
import numpy as np


t = np.linspace(0.,1., 128)

gamma = ex.gamma_example("sine")[0](t)
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.set_ylim([0, 1])
ax.set_xlim([0, 1])
ax.set_title("Gamma Curve")
ax.set_ylabel("Error")
ax.set_xlabel("n (number of nodes)")
plt.plot(t, gamma)
plt.plot(t, gamma + 0.05)
plt.plot(t, gamma-0.05)
plt.show()