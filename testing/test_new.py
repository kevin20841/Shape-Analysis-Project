print("Loading.....")
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from shapedist.elastic_linear_old import find_gamma, find_error
from testing import examples as ex
import numpy as np
print("Calculating......")
m = 131
n = 131
t = np.linspace(0.,1., m)

p = [0, 0]

q = ex.curve_example('bumps', t)[0]

x_function = InterpolatedUnivariateSpline(t, q[0])
y_function = InterpolatedUnivariateSpline(t, q[1])




test = ex.gamma_example("identity")[0](t)
test1 = ex.gamma_example("identity")[0](t)
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

num = int(np.log2(m) - 4)

print(num)

parametrization_array = np.random.rand(3, m)

parametrization_array = np.array(
    [[True, False, False, False, True, False, False, False, True, False, False, False, True, False, False, True,
      True, False, False, False, True, False, False, False, True, False, False, False, True, False, False, True],
        [True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, True,
      True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, True],
        [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
         True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]]

)
num_true = [4,6,7,8]
i = 0
j = 0


tg1, gammay, energy_1 = find_gamma(np.array([t, p[1]]), np.array([t, q[1]]), 12, 16, num)
print("hi")


#tg2, gammax, energy_2 = find_gamma(np.array([t, p[0]]), np.array([t, q[0]]), 6, 8, num)
# tg1, gammay, energy_1 =find_gamma(np.array([t, p[1]]), np.array([t, q[1]]),  np.array([t, t]), 12, 16)
# tg2, gammax, energy_2 =find_gamma(np.array([t, p[0]]), np.array([t, q[0]]), np.array([t, t]), 12, 16)

print("Minimum Energies:")
#print("x:", energy_1)
# print("y:", energy_2)
print("Errors:")

# print("x:", find_error(domain_x, ex.gamma_example("sine")[0](domain_x), gammax))
# print("y:", find_error(domain_y, ex.gamma_example("sine")[0](domain_y), gammay))
print("Finished!")

plt.plot(p[0], p[1], '.-b')
plt.plot(q[0], q[1], '.r')
plt.figure()
#plt.plot(x_function(gammax), y_function(gammay), ".-")
plt.plot(tg1, gammay, ".-b")
plt.plot(t, test1, "-r")

# plt.plot(tg1, gammay - test1, "-b")

print( np.sqrt( 1/(m-1) * np.sum( (gammay - test1)**2 ) ) )
print(find_error(tg1, gammay, test1))
print(find_error(tg1, ex.gamma_example("identity")[0](t), gammay))
plt.show()
