import sys
sys.path.append("../")
import time
import scipy.interpolate
import shapedist
from testing import examples as ex
import numpy as np


def run_tests(ndim, test_cases, test_sizes, n_iter):
    for case in test_cases:
        print("Curve type in {} dimensions:".format(str(ndim)), case)
        print("{: <15}| {: <15}| {: <15}| {: <15}".format("Size", "Energy", "Error", "Time(seconds)"))
        for size in test_sizes:
            n = size  # number of points in domain
            
            t = np.linspace(0., 1., n)
            gamma_sol = ex.gamma_example(case)[0](t)
            q = np.random.rand(n, ndim)
            p = np.zeros(q.shape)
            for d in range(ndim):
                inter_func = scipy.interpolate.CubicSpline(t, q[:, d])
                p[:, d] = inter_func(gamma_sol)
            elapsed = np.inf
            energy = 0
            tg = 0
            gammay = 0
            for i in range(n_iter):
                start = time.time()
                energy, p_new, q_new, tg, gammay = shapedist.find_shapedist(p, q, 'ud', t1=t, t2=t)
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
    ndim = 5
    if len(sys.argv) == 5:
        ndim = int(sys.argv[1])
        test_cases = sys.argv[2].split(" ")
        test_sizes = [int(x) for x in sys.argv[3].split(" ")]
        n_iter = int(sys.argv[4])
    elif len(sys.argv) == 1:
        pass
    elif len(sys.argv) == 2:
        ndim = int(sys.argv[1])
    else:
        raise RuntimeWarning("Incorrect arguments passed!")
    run_tests(ndim, test_cases, test_sizes, n_iter)


if __name__ == "__main__":
    sys.exit(main())
