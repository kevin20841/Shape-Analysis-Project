import matplotlib.pyplot as plt
from elasticity_n_4 import *
import examples as ex
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import interp1d
m = 80
n = 200
t = np.linspace(0.,1., m)

p = [0,0]

q = ex.curve_example('bumps', t)[0]

x_function = InterpolatedUnivariateSpline(t, q[0])
y_function = InterpolatedUnivariateSpline(t, q[1])

p[0] = x_function(ex.gamma_example("steep")[0](t))
p[1] = y_function(ex.gamma_example("steep")[0](t))

test = ex.gamma_example("steep")[0](t)
i = 1
while i < len(test):
    if (test[i]-test[i-1]) / (t[i]-t[i-1]) < 0:
        print("BAD THINGS")
    i = i + 1


# p = np.array(np.sin(t))
# q = np.array(np.exp(t)-1)
tg = np.linspace(0.,1.,n)
gamma = np.linspace(0., 1., n)

gammay = find_gamma(t, t, p[1], q[1], tg, gamma, InterpolatedUnivariateSpline)

gammax = find_gamma(t, t, p[0], q[0], tg, gamma, InterpolatedUnivariateSpline)


plt.plot(p[0], p[1], 'b')
plt.plot(q[0], q[1], 'g')
plt.plot(x_function(gammax), y_function(gammay), ".-")
plt.plot(t, gammax, ".-r")
plt.plot(t, ex.gamma_example("steep")[0](t), ".-y")

plt.show()

