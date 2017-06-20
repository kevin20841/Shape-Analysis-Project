import matplotlib.pyplot as plt
from elasticity_n_4 import *
import examples as ex

n = 50
t = np.linspace(0.,1., n)

g = ex.curve_example('bumps', t)[0]


p = g

g = ex.curve_example('circle', t)[0]
q = g

p = p[0]
q = q[0]

p = -p + p[0]
p = np.abs(p)
q = -q + q[0]
#
# p = np.array(np.sin(t))
# q = np.array(np.exp(t)-1)
tg = np.linspace(0.,1.,n)
gamma = np.linspace(0., 1., n)

gamma = find_gamma(t, t, p, q, tg, gamma)


print(find_error(tg, gamma, np.zeros(n)))
g, g_deriv = ex.gamma_example('sine',0.05)

x = np.interp(gamma, t, q)
plt.plot(t, p, 'b')
plt.plot(t, q, 'g')
plt.plot(t, x, 'y')
plt.plot(tg, gamma, "r")
plt.show()

