import examples as ex
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

n = 128
t = np.linspace(0.,1.,n)

b, b_deriv = ex.curve_example('hippopede',n)
g, g_deriv = ex.gamma_example('sine',0.05)
plt.plot(t,g(t))

q = b[0]
q_function = InterpolatedUnivariateSpline(t,q)
p = q_function( g(t) )
plt.plot(t,p,t,q)
plt.show()
