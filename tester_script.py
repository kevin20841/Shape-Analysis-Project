import cProfile
import pstats
from shapedist import elastic_linear
from io import StringIO
import numpy as np
from shapedist.elastic_linear import *
import examples as ex
from scipy.interpolate import InterpolatedUnivariateSpline

readFile = open("test_cases", "r")
text = readFile.read()
readFile.close()
text = text.split("\n")
f = open("test_results_raw.txt", "w")

for i in text:
    x = i.split(" ")
    m = int(x[1])
    curve_type = str(x[0])

    print(m, curve_type)
    t = np.linspace(0., 1., m)
    q = ex.curve_example('bumps', t)[0][0]


    x_function = InterpolatedUnivariateSpline(t, q)

    test = ex.gamma_example(curve_type)[0](t)

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

    p = x_function(test)

    # p = np.array(np.sin(t))
    # q = np.array(np.exp(t)-1)
    tg = np.linspace(0.,1., m)
    gamma = np.linspace(0., 1., m)

    domain_x, gammax, val = elastic_linear.find_gamma(np.array([t, p]), np.array([t, q]), -1, -1)
    error = find_error(domain_x, ex.gamma_example(curve_type)[0](domain_x), gammax)
    f.write("Minimum Energy: " + str(val) + " Error: " + str(error) + "\n\n\n")
    for j in range(5):

        cProfile.run("elastic_linear.find_gamma(np.array([t, p]), np.array([t, q]), -1, -1)", 'statsfile')

        stream = StringIO()
        stats = pstats.Stats('statsfile', stream=stream).sort_stats("cumulative")
        stats.print_stats()

        f.write(stream.getvalue())

f.close()

