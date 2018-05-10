print("Loading.....")
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from shapedist.elastic_linear import find_gamma
from testing import examples as ex
import numpy as np
print("Calculating......")
m = 256
n = 256
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
tg = np.linspace(0.,1., 64)
gamma = np.linspace(0., 1., 64)


tg1, gammay, energy_1 = find_gamma(np.array([t, p[1]]), np.array([t, q[1]]), 12, 16, 6)
print("hi")
tg2, gammax, energy_2 = find_gamma(np.array([t, p[0]]), np.array([t, q[0]]), 12, 16, 6)



print("Minimum Energies:")
print("x:", energy_1)
print("y:", energy_2)
print("Errors:")
#print("x:", find_error(domain_x, ex.gamma_example("sine")[0](domain_x), gammax))
#print("y:", find_error(domain_y, ex.gamma_example("sine")[0](domain_y), gammay))
print("Finished!")

plt.plot(p[0], p[1], '.k')
plt.plot(q[0], q[1], '.g')

#plt.plot(x_function(gammax), y_function(gammay), ".-")
plt.plot(tg1, gammay, ".-r")
plt.plot(t, test1, ".-y")

plt.show()
