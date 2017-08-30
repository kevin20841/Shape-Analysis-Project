import matplotlib.pyplot as plt
import numpy as np

from testing import examples as ex

n = 128
t = np.linspace(0.,1.,n)

g = ex.curve_example('bumps', t)[0]
print(len(g[0]))

plt.plot(t,g[0])
plt.show()