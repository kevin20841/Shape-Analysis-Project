import cProfile
import pstats
from io import StringIO

from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
from shapedist_old import elastic_n_2
from testing import examples as ex

readFile = open("test_cases_n_2", "r")
text = readFile.read()
readFile.close()
text = text.split("\n")
f = open("test_results_n_2_raw.txt", "w")

for i in text:
    x = i.split(" ")
    m = int(x[1])
    height1 = int(x[2])
    height2 = int(x[3])
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

    domain_x, gammax, val = elastic_n_2.find_gamma(np.array([t, p]), np.array([t, q]), np.array([tg, gamma]),
                                                   height1, height2)
    error = elastic_n_2.find_error(domain_x, ex.gamma_example(curve_type)[0](domain_x), gammax)
    f.write("Minimum Energy: " + str(val) + " Error: " + str(error) +  " " + str(height1) + " " + str(height2)
            + "\n\n\n")
    for j in range(5):
        cProfile.run("elastic_n_2.find_gamma(np.array([t, p]), np.array([t, q]), np.array([tg, gamma]),  height1, height2)", 'statsfile')
        stream = StringIO()
        stats = pstats.Stats('statsfile', stream=stream).sort_stats("cumulative")
        stats.print_stats()

        f.write(stream.getvalue())

f.close()

