import shapedist
import numpy as np
from numba import jit, float64
import warnings
import matplotlib.pyplot as plt


def elastic_matcher(p, q, dim, parametrization=None, curve_type="coord", gamma_tol=0.0001, adaptive=True, hierarchy_tol=1, n_levels=3, max_iter=5, hierarchy_factor=2,
                    energy_dot=False, interpolation_method='linear'):
    t1 = None
    t2 = None
    if not(parametrization is None):
        t1 = parametrization[0]
        t2 = parametrization[1]
    if dim == 1 and adaptive:
        if hierarchy_tol == 1:
            hierarchy_tol = 2e-5
        original, boolean_mask, curve_hierarchy = \
            shapedist.build_hierarchy_1D.hierarchical_curve_discretization(np.array([p, q]), hierarchy_tol,
                                                                           n_levels=n_levels, max_iter=max_iter,
                                                                           interpolation_method=interpolation_method)

        t_orig = original[0]

        b1_orig = original[1]
        b2_orig = original[2]
        for i in boolean_mask:
            print(t_orig[i].size)
        #     plt.figure()
        #     plt.plot(t_orig[i], b1_orig[i], ".-r")
        #     plt.plot(t_orig[i], b2_orig[i], ".-b")
        # plt.show()
        if t_orig[boolean_mask[-1]].size > 500:
            warnings.warn("Algorithm will run slowly because curves are not coarsened enough. A larger hierarchy tolerance is recommended.", RuntimeWarning)
        tg, gamma, energy = shapedist.elastic_linear_hierarchy.find_gamma(t_orig, b1_orig, b2_orig, boolean_mask,
                                                                              energy_dot, gamma_tol)
        return tg, gamma, energy

    elif dim == 2 and adaptive:
        if hierarchy_tol == 1:
            hierarchy_tol = 2e-3
        original, boolean_mask, curve_hierarchy = \
            shapedist.build_hierarchy_2D.hierarchical_curve_discretization(np.array([p, q]),
                                                                           t1=t1, t2=t2,
                                                                           init_coarsening_tol=hierarchy_tol,
                                                                           n_levels=n_levels, max_iter=max_iter,
                                                                           interpolation_method=interpolation_method,
                                                                           curve_type=curve_type)

        t_orig = original[0]
        b1_orig = original[1]
        b2_orig = original[2]
        for i in boolean_mask:
            print(t_orig[i].size)
        if t_orig[boolean_mask[-1]].size > 500:
            warnings.warn("Algorithm will run slowly because curves are not coarsened enough. A larger hierarchy tolerance is recommended.", RuntimeWarning)
        tg, gammay, energy = shapedist.elastic_linear_hierarchy.find_gamma(t_orig, b1_orig, b2_orig, boolean_mask,
                                                                           energy_dot, gamma_tol)

        return tg, gammay, energy


@jit(float64(float64[:], float64[:], float64[:]), cache=True, nopython=True)
def find_error(tg, gammar, gammat):
    n = tg.size
    error = 1 / 2 * (tg[1] - tg[0]) * (gammar[1] - gammat[1]) ** 2 + 1 / 2 * (tg[n-1] - tg[n-2]) * (gammar[n-1] - gammat[n-1]) ** 2
    k = 2
    if n != gammar.size or n != gammat.size:
        raise IndexError
    while k < n-1:
        error = error + 1/2 * (gammar[k] - gammat[k]) ** 2 * (tg[k] - tg[k-1]) ** 2
        k = k + 1
    error = error ** (1/2)
    return error
