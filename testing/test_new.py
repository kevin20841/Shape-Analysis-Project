print("Loading.....")
import matplotlib.pyplot as plt
import scipy.interpolate
import shapedist
from testing import examples as ex
import numpy as np
import cProfile
import pstats
from io import StringIO

print("Calculating......")
m = 2111
n = 2111
t = np.linspace(0.,1., m)
# t = np.random.uniform(0, 1, m-2)
# t = np.append(t, 1)
# t = np.insert(t, 0, 0)
# t.sort()

t2 = np.linspace(0., 1., n)
# t_random =np.random.rand(m)
# t_random.sort()
# t_random[0] = 0
# t_random[-1] = 1
p = [0, 0]

q = ex.curve_example('bumps', t)[0]

x_function = scipy.interpolate.CubicSpline(t, q[0])
y_function = scipy.interpolate.CubicSpline(t, q[1])


test = ex.gamma_example("bumpy")[0](t)
test1 = ex.gamma_example("bumpy")[0](t)
print(t)
print(1/31)
p[0] = x_function(test)
p[1] = y_function(test)
# p = ex.curve_example('hippopede', t2)[0]
p = np.array(p)
q = np.array(q)

x = np.zeros([n, 2])
y = np.zeros([m, 2])
R_theta = np.array([[0, 1], [-1, 0]])
print(R_theta)
# p = np.matmul(R_theta, p)
x[:, 0] = p[0]
x[:, 1] = p[1]
y[:, 0] = q[0]
y[:, 1] = q[1]
# print(x.shape, y.shape, t.shape, t2.shape)
plt.plot(t, test, ".-b")
plt.show()
# tg, gammay, energy = shapedist.elastic_matcher(x, y, dim=2, curve_type="coords", energy_dot=False)
tg, gammay, energy = shapedist.elastic_matcher(np.array([t, p[1]]), np.array([t, q[1]]), 1)
# tg, gammay, energy = shapedist.elastic_n_2.find_gamma(t, x, y, t, t, 4, 4)
# print(shapedist.find_error(t, test, gammay))
plt.plot(t, test, ".-b")
plt.plot(tg, gammay, ".-r")

plt.show()


#
# plt.show()
cProfile.runctx('tg, gammay, energy = shapedist.elastic_matcher(np.array([t, p[1]]), np.array([t, q[1]]), 1)',
                globals(),
                locals(), filename="statsfile")
stream = StringIO()
stats = pstats.Stats('statsfile', stream=stream).sort_stats("cumulative")
stats.print_stats()
print(stream.getvalue())
