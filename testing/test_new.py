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
m = 200
n = 200
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


test = ex.gamma_example("sine")[0](t)
# test1 = ex.gamma_example("sine")[0](t)

print(np.gradient(test, t).min())
p[0] = x_function(test)
p[1] = y_function(test)

p = np.array(p)
q = np.array(q)
# p = ex.curve_example('hippopede', t2)[0]
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
tg, gammay, energy, original, boolean_mask = shapedist.elastic_matcher(
    np.array([t, p[0]]), np.array([t, q[0]]), dim=1, curve_type="normals",
                                               adaptive=True, energy_dot=False)

# tg, gammay, energy = shapedist.elastic_matcher(np.array([t, p[1]]), np.array([t, q[1]]), 1, adaptive=False)
# tg, gammay, energy = shapedist.elastic_n_2.find_gamma(t, p[0], q[0], t, t, 4, 4, False)
# print(shapedist.find_error(tg, ex.gamma_example("bumpy")[0](tg), gammay))
plt.plot(tg, gammay, ".-r")

plt.show()
#
# cProfile.runctx('tg, gammay, energy, boolean_mask = shapedist.elastic_matcher(x, y, parametrization=[t, t], dim=2, curve_type="SRVF",adaptive=False, energy_dot=True)',
#                 globals(),
#                 locals(), filename="statsfile")
# stream = StringIO()
# stats = pstats.Stats('statsfile', stream=stream).sort_stats("cumulative")
# stats.print_stats()
# print(stream.getvalue())
