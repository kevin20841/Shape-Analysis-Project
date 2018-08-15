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
m = 400
n = 300
t = np.linspace(0.,1., m)
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
test1 = ex.gamma_example("sine")[0](t)

p[0] = x_function(test)
p[1] = y_function(test1)
p = np.array(p)
q = np.array(q)
x = np.zeros([m, 2])
y = np.zeros([m, 2])


x[:, 0] = p[0]
x[:, 1] = p[1]
y[:, 0] = q[0]
y[:, 1] = q[1]


tg, gammay, energy = shapedist.elastic_matcher(x, y, dim=2, curve_type="normals")
tg, gammay, energy = shapedist.elastic_matcher(np.array([t, p[1]]), np.array([t, q[1]]), 1)
plt.plot(tg, gammay, ".-r")
plt.plot(t, test, ".-b")
plt.show()

# cProfile.runctx('tg, gammay, energy = shapedist.elastic_matcher(x,y, dim=2)',
#                             globals(),
#                             locals(), filename="statsfile")
# stream = StringIO()
# stats = pstats.Stats('statsfile', stream=stream).sort_stats("cumulative")
# stats.print_stats()
# print(stream.getvalue())
