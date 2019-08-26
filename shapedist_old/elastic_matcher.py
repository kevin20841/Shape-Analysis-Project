import shapedist
import numpy as np
from numba import jit, float64
import matplotlib.pyplot as plt


def elastic_matcher(p, q, dim, curve_type="coord", adaptive=True, hierarchy_tol=2e-5, n_levels=3, max_iter=5, hierarchy_factor=2,
                    energy_dot=False, interpolation_method='linear'):
    if dim == 1 and adaptive:
        original, boolean_mask, curve_hierarchy = \
            shapedist.build_hierarchy_1D.hierarchical_curve_discretization(np.array([p, q]), hierarchy_tol,
                                                                           n_levels=n_levels, max_iter=max_iter,
                                                                           hierarchy_factor=hierarchy_factor,
                                                                           interpolation_method=interpolation_method)

        t_orig = original[0]
        b1_orig = original[1]
        b2_orig = original[2]
        tg, gamma, energy = shapedist.elastic_linear_reduced.find_gamma(t_orig, b1_orig, b2_orig, boolean_mask,
                                                                        energy_dot,,
        return tg, gamma, energy

    elif dim == 2 and adaptive:

        original, boolean_mask, curve_hierarchy = \
            shapedist.build_hierarchy_2D.hierarchical_curve_discretization(np.array([p, q]), hierarchy_tol,
                                                                           n_levels=n_levels, max_iter=max_iter,
                                                                           hierarchy_factor=hierarchy_factor,
                                                                           interpolation_method=interpolation_method,
                                                                           curve_type=curve_type)

        t_orig = original[0]
        b1_orig = original[1]
        b2_orig = original[2]
        for i in boolean_mask:
            print(t_orig[i].size)
        tg, gammay, energy = shapedist.elastic_linear_reduced.find_gamma(t_orig, b1_orig, b2_orig, boolean_mask,
                                                                         energy_dot,,

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
