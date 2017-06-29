print("Loading.....")
#import matplotlib.pyplot as plt
from shapedist.elastic_linear import *
import examples as ex
from scipy.interpolate import InterpolatedUnivariateSpline
print("Calculating......")
m = 2048
n = 1024
t = np.linspace(0.,1., m)

p = [0, 0]

q = ex.curve_example('bumps', t)[0]

x_function = InterpolatedUnivariateSpline(t, q[0])
y_function = InterpolatedUnivariateSpline(t, q[1])


test = ex.gamma_example("sine")[0](t)
test1 = ex.gamma_example("sine")[0](t)
#
# test = np.zeros(m)
#
# i = 1
# while i < m:
#     test[i] = np.random.random_sample()
#     i = i + 1
# test.sort()
# test[m-1] = 1
# test[0] = 0

p[0] = x_function(test)
p[1] = y_function(test1)

# p = np.array(np.sin(t))
# q = np.array(np.exp(t)-1)
tg = np.linspace(0.,1.,n)
gamma = np.linspace(0., 1., n)

domain_y, gammay, miney = find_gamma(np.array([t, p[1]]), np.array([t, q[1]]), 100, 8)

domain_x, gammax, minex = find_gamma(np.array([t, p[0]]), np.array([t, q[0]]), 100, 8)


print("Minimum Energies:")
print("x:", miney)
print("y:", minex)
print("Errors:")
#print("x:", find_error(domain_x, ex.gamma_example("sine")[0](domain_x), gammax))
#print("y:", find_error(domain_y, ex.gamma_example("sine")[0](domain_y), gammay))
print("Finished!")

# plt.plot(p[0], p[1], 'k')
# plt.plot(q[0], q[1], 'g')
#
# plt.plot(x_function(gammax), y_function(gammay), ".-")
# plt.plot(domain_y, gammay, ".-r")
# plt.plot(t, test1, ".-y")
# plt.plot(tg, gammax, "-c")
# plt.plot(t, test, "-m")
#
# plt.show()

