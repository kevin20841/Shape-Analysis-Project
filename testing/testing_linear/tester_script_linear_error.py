import cProfile
import pstats
from io import StringIO

from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
from shapedist_old.elastic_linear_old_hierarchy import find_gamma, find_error
from testing import examples as ex
import matplotlib.pyplot as plt
def calc(filename):
    readFile = open(filename, "r")
    text = readFile.read()
    readFile.close()
    text = text.split("\n")
    f = open("errors.csv", "a")
    f.write("\n")
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
        f.write(str(m) + ",")
        i = 9
        while i <= 9:
            x = int(np.log2(t.size) - 4)
            domain_x, gammax, val = find_gamma(np.array([t, p]), np.array([t, q]), 6, 8, x)

            error = find_error(domain_x, ex.gamma_example(curve_type)[0](domain_x), gammax)
            f.write(str(error) + ",")
            i = i + 1
            print(error, x)
            plt.plot(domain_x, gammax - ex.gamma_example(curve_type)[0](domain_x), "-b")
            plt.plot(domain_x, gammax, ".-b")
            plt.plot(domain_x, ex.gamma_example(curve_type)[0](domain_x), "-r")
            plt.show()
        f.write("\n")

    f.close()

calc("test_case_i")
calc("test_case_s")
calc("test_case_b")