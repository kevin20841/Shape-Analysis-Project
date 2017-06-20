#import matplotlib.pyplot as plt
from elasticity_n_4 import *
import examples as ex

n = 130
t = np.linspace(0.,1., n)
t1 = np.zeros(n)
p = np.array(np.sin(t))
q = np.array(np.exp(t)-1)
tg = np.linspace(0.,1.,130)
gamma = np.linspace(0., 1., 130)

gamma = find_gamma(t, t, p, q, tg, gamma)

g, g_deriv = ex.gamma_example('sine',0.05)

x = np.interp(gamma, t, q)
# plt.plot(t, p, 'b')
# plt.plot(t, q, 'g')
# plt.plot(t, x, 'y')
# plt.plot(tg, gamma, "r")
# plt.show()

