import matplotlib.pyplot as plt
import numpy as np
from elasticity_n_4 import *
import examples as ex

n = 17
t = np.linspace(0.,1., n)
t1 = np.zeros(n)
p = np.array(t**4)
q = np.array(t)
tg = np.linspace(0.,1.,20)
gamma = np.linspace(0., 1., 20)

gamma = find_gamma(t, t, p, q, tg, gamma)

print(gamma)
g, g_deriv = ex.gamma_example('sine',0.05)

plt.plot(t, p)
plt.plot(t, q)
plt.plot(tg, gamma, "r")
plt.show()

