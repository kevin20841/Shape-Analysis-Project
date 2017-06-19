import matplotlib.pyplot as plt
import numpy as np
from elasticity_n_4 import *
import examples as ex

n = 5
t = np.linspace(0.,1., n)
t1 = np.zeros(n)
p = np.array(np.sin(t))
q = np.array(np.sin(t))

print(len(p), len(q))
tg = np.linspace(0.,1.,10)
gamma = np.linspace(0., 1., 10)

gamma = [0] + find_gamma(t, t, p, q, tg, gamma) + [1]

print(gamma)
g, g_deriv = ex.gamma_example('sine',0.05)

plt.plot(t, p)
plt.plot(t, q)
plt.plot(tg, gamma, "r")
plt.show()

