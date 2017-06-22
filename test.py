print("Loading.....")
import matplotlib.pyplot as plt
from shapedist.elastic_global import *
import examples as ex
from scipy.interpolate import InterpolatedUnivariateSpline
print("Calculating......")
m = 80
n = 160
t = np.linspace(0.,1., m)

p = [0, 0]

q = ex.curve_example('flower', t)[0]

x_function = InterpolatedUnivariateSpline(t, q[0])
y_function = InterpolatedUnivariateSpline(t, q[1])


test = ex.gamma_example("sine")[0](t)
test1 = ex.gamma_example("polynomial")[0](t)
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

gammay, gammayE = find_gamma([t, p[1]], [t, q[1]], [tg, gamma])

gammax, gammaxE = find_gamma([t, p[0]], [t, q[0]], [tg, gamma])

print("Minimum Energies:")
print("x:", gammaxE)
print("y:", gammayE)
print("Error x:", find_error(t, test, gammax))
print("Error y:", find_error(t, test1, gammay))
print("Finished!")
plt.plot(p[0]- 0.5, p[1], 'k')
plt.plot(q[0] - 0.25, q[1], 'g')
plt.plot(x_function(gammax), y_function(gammay), ".-")
plt.plot(t, gammay, "-r")
plt.plot(t, test1, "-y")
plt.plot(t, gammax, "-c")
plt.plot(t, test, "-m")

plt.show()

