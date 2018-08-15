import cProfile
import pstats
from io import StringIO
from scipy.interpolate import CubicSpline
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
import shapedist
from testing import examples as ex
import matplotlib.pyplot as plt
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
        #print(m)
        t = np.linspace(0., 1., m)
        q = ex.curve_example('bumps', t)[0][1]

        x_function = CubicSpline(t, q)
        test = ex.gamma_example(curve_type)[0](t)
        p = x_function(test)
        original, boolean_mask, curve_hierarchy = \
            shapedist.build_hierarchy_1D.hierarchical_curve_discretization(np.array([[t, p], [t, q]]), 2e-5)

        t_orig = original[0]
        b1_orig = original[1]
        b2_orig = original[2]

        print(t_orig[boolean_mask[-1]].size)
        domain_x, gammax, val = shapedist.elastic_linear_hierarchy.find_gamma(t_orig, b1_orig, b2_orig, boolean_mask, 8, 6)
        error = shapedist.find_error(domain_x, ex.gamma_example(curve_type)[0](domain_x), gammax)
        #print(error)
        f.write("Minimum Energy: " + str(val) + " Error: " + str(error) + "\n\n\n")
        for j in range(3):
            cProfile.runctx('tg1, gammay, energy_1 = shapedist.elastic_linear_hierarchy.find_gamma(t_orig, b1_orig, b2_orig, boolean_mask, 8, 6)',
                            globals(),
                            locals(), filename="statsfile")
            stream = StringIO()
            stats = pstats.Stats('statsfile', stream=stream).sort_stats("cumulative")
            stats.print_stats()
            f.write(stream.getvalue())
    f.close()

calc("test_case_i")
calc("test_case_s")
calc("test_case_b")