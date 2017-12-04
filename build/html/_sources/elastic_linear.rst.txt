Linear Elasticity Algorithm
===========================

.. toctree::
   :maxdepth: 2

.. automodule:: shapedist.elastic_linear

.. autofunction:: shapedist.elastic_linear.interp()

.. autofunction:: shapedist.elastic_linear.find_gamma()

.. autofunction:: shapedist.elastic_linear.find_error()

Examples:
---------

Below is an example. We construct an artificial gamma, and use p = q(gamma(t)) and q(t) in order to find a calculated
gamma(t). The calculated gamma(t) is plotted and compared to the artificial gamma(t). If they are the same, the algorithm
is functioning correctly.

.. code-block:: python

   print("Loading.....")
   import matplotlib.pyplot as plt
   from scipy.interpolate import InterpolatedUnivariateSpline

   from shapedist.elastic_linear import *
   from testing import examples as ex

   print("Calculating......")
   m = 128
   n = 128
   t = np.linspace(0.,1., m)

   p = [0, 0]

   q = ex.curve_example('bumps', t)[0]

   x_function = InterpolatedUnivariateSpline(t, q[0])
   y_function = InterpolatedUnivariateSpline(t, q[1])


   test = ex.gamma_example("steep")[0](t)
   test1 = ex.gamma_example("steep")[0](t)

   p[0] = x_function(test)
   p[1] = y_function(test1)

   tg = np.linspace(0.,1.,n)
   gamma = np.linspace(0., 1., n)

   domain_y, gammay, miney = find_gamma(np.array([t, p[1]]), np.array([t, q[1]]), -1, -1)

   domain_x, gammax, minex = find_gamma(np.array([t, p[0]]), np.array([t, q[0]]), -1, -1)


   print("Minimum Energies:")
   print("x:", miney)
   print("y:", minex)

   print("Finished!")

   plt.plot(p[0], p[1], '.-k')
   plt.plot(q[0], q[1], '.-g')

   plt.plot(x_function(gammax), y_function(gammay), ".-")
   plt.plot(domain_y, gammay, ".-r")
   plt.plot(t, test1, ".-y")

   plt.show()

