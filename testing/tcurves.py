import sys

sys.path.append("../")
import time
import scipy.interpolate
import shapedist
from testing import examples as ex
import numpy as np


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    print(*args)


def run_tests(ndim, test_cases, curve_name, test_sizes, n_iter, shape_reps, mode):
    mode = mode + "d"
    for shape_rep in shape_reps:
        for case in test_cases:
            eprint(curve_name.upper() + " curve, " + case.upper() + " gamma, in {} dimensions with ".format(str(ndim)) + shape_rep.__name__ + " representation")
            eprint("{: <15}| {: <15}| {: <15}| {: <15}| {: <15}".format("Size", "Coarsened Size", "Energy", "Error", "Time(seconds)"))
            for size in test_sizes:
                n = size  # number of points in domain

                t = np.linspace(0., 1., n)
                gamma_sol = ex.gamma_example(case)[0](t)
                if curve_name == "rand":
                    q = np.random.rand(n, ndim)
                else:
                    if ndim > 2:
                        raise RuntimeWarning(
                            "There are no synthetic curves larger than 2 dimensions currently implemented")
                    q = ex.curve_example(curve_name, t)[0][:ndim].T
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
                    energy, p_new, q_new, tg, gammay = shapedist.find_shapedist(p, q, mode, t1=t, t2=t, shape_rep=shape_rep)
                    end = time.time()
                    if start - end < elapsed:
                        elapsed = end - start

                error = shapedist.find_error(tg, gammay, ex.gamma_example(case)[0](tg))
                eprint("{: <15}| {: <15}| {: <15}| {: <15}| {: <15}".format(str(size),str(tg.shape[0]), "%0.10f" % energy, "%0.10f" % error,

                                                                  "%0.10f" % elapsed))
        eprint()


def main():
    # test_cases = ["identity", "sine", "flat", "steep", "bumpy"]
    test_cases = ["sine", "bumpy"]
    test_sizes = [256, 512, 1024, 2048, 4096]
    # test_sizes= [i for i in range(60, 200, 10)]
    curve_name = "limacon"
    n_iter = 5
    # 1d test
    # shape_reps = [shapedist.coords]
    # run_tests(1, test_cases, curve_name, test_sizes, n_iter, shape_reps)
    
    # 2d test
    shape_reps = [shapedist.coords, shapedist.tangent, shapedist.curvature, shapedist.srvf]

    # shape_reps = [shapedist.curvature]
    run_tests(2, test_cases, curve_name, test_sizes, n_iter, shape_reps, "um")
    
    # 3d test
    # shape_reps = [shapedist.coords]
    # runtests(3, test_cases, curve_name, test_sizes, n_iter, shape_reps)

if __name__ == "__main__":
    sys.exit(main())
