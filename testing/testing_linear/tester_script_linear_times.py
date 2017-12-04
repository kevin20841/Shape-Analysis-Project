import cProfile
import pstats
from io import StringIO

from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
from shapedist import elastic_linear
from testing import examples as ex
def calc(filename):
    readFile = open(filename, "r")
    text = readFile.read()
    readFile.close()
    text = text.split("\n")
    f = open(filename.split("_")[len(filename.split("_"))-1] + "_raw.txt", "w")

    for i in text:
        x = i.split(" ")

        m = int(x[1])

        curve_type = str(x[0])
        print(m)
        t = np.linspace(0., 1., m)
        # t = np.random.rand(m)
        #
        # t[0] = 0
        # t[t.size - 1] = 1
        # t.sort()
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

        domain_x, gammax, val = elastic_linear.find_gamma(np.array([t, p]), np.array([t, q]), 6, 12, 6)
        error = elastic_linear.find_error(domain_x, ex.gamma_example(curve_type)[0](domain_x), gammax)
        f.write("Minimum Energy: " + str(val) + " Error: " + str(error) + "\n\n\n")
        for j in range(3):
            cProfile.runctx("elastic_linear.find_gamma(np.array([t, p]), np.array([t, q]), 6, 12, 6)", globals(),
                            locals(), filename="statsfile")
            stream = StringIO()
            stats = pstats.Stats('statsfile', stream=stream).sort_stats("cumulative")
            stats.print_stats()

            f.write(stream.getvalue())

    f.close()

calc("test_case_i")
calc("test_case_s")
calc("test_case_b")