import sys
sys.path.append("../")
import time
import scipy.interpolate
import shapedist
from testing import examples as ex
import numpy as np


def run_tests(test_cases, test_sizes, n_iter):
    for case in test_cases:
        print("Curve type:", case)
        print("{: <15}| {: <15}| {: <15}| {: <15}".format("Size", "Energy", "Error", "Time(seconds)"))
        for size in test_sizes:
            n = size  # number of points in domain

            t = np.linspace(0., 1., n)
            gamma_sol = ex.gamma_example(case)[0](t)
            q = ex.curve_example("circle", t)[0]

            inter_func1 = scipy.interpolate.CubicSpline(t, q[0])
            inter_func2 = scipy.interpolate.CubicSpline(t, q[1])
            p = np.zeros(q.shape)
            p[0] = inter_func1(gamma_sol)
            p[1] = inter_func2(gamma_sol)
            p = p.T
            q = q.T
            elapsed = np.inf
            energy = 0
            tg = 0
            gammay = 0
            for i in range(n_iter):
                start = time.time()
                energy, p_new, q_new, tg, gammay = shapedist.find_shapedist(p, q, 'ud', t1=t, t2=t, shape_rep=shapedist.srvf)
                end = time.time()

                if start-end <np.inf:
                    elapsed = end-start

            error = shapedist.find_error(tg, gammay, ex.gamma_example("bumpy")[0](tg))
            print("{: <15}| {: <15}| {: <15}| {: <15}".format(str(size), "%0.10f" % energy, "%0.10f" % error,
                                                           "%0.10f" % elapsed))


def main():
    test_cases = ["identity", "sine", "flat", "steep", "bumpy"]
    # test_sizes = [256, 512, 1024, 2048]
    test_sizes = [i for i in range(200, 2200, 200)]
    n_iter = 5
    if len(sys.argv) == 4:
        test_cases = sys.argv[1].split(" ")
        test_sizes = [int(x) for x in sys.argv[2].split(" ")]
        n_iter = int(sys.argv[3])
    elif len(sys.argv) == 1:
        pass
    else:
        raise RuntimeWarning("Incorrect arguments passed!")
    run_tests(test_cases, test_sizes, n_iter)


if __name__ == "__main__":
    sys.exit(main())
