import shapedist
import numpy as np
from numba import jit, float64
import warnings
from math import pi
import matplotlib.pyplot as plt


# def elastic_matcher(p, q, dim, parametrization=None, curve_type="coord", gamma_tol=0.0001, adaptive=True, hierarchy_tol=1, n_levels=3, max_iter=5, hierarchy_factor=2,
#                     energy_dot=False, interpolation_method='linear'):
#     t1 = None
#     t2 = None
#     if not(parametrization is None):
#         t1 = parametrization[0]
#         t2 = parametrization[1]
#     if dim == 1:
#         if hierarchy_tol == 1:
#             hierarchy_tol = 2e-5
#         original, boolean_mask, curve_hierarchy = \
#             shapedist.build_hierarchy_1D.hierarchical_curve_discretization(np.array([p, q]), hierarchy_tol,
#                                                                            n_levels=n_levels, max_iter=max_iter,
#                                                                            adaptive = adaptive,
#                                                                            interpolation_method=interpolation_method)
#
#         t_orig = original[0]
#
#         b1_orig = original[1]
#         b2_orig = original[2]
#         #     plt.figure()
#         #     plt.plot(t_orig[i], b1_orig[i], ".-r")
#         #     plt.plot(t_orig[i], b2_orig[i], ".-b")
#         # plt.show()
#         if t_orig[boolean_mask[1]].size > 500:
#             warnings.warn("Algorithm will run slowly because curves are not coarsened enough."
#                           " A larger hierarchy tolerance is recommended.", RuntimeWarning)
#         tg, gamma, energy = shapedist.elastic_linear_hierarchy.find_gamma(t_orig, b1_orig, b2_orig, boolean_mask,
#                                                                               energy_dot, gamma_tol)
#         return tg, gamma, energy, original, boolean_mask
#
#     elif dim == 2:
#         if hierarchy_tol == 1:
#             hierarchy_tol = 2e-3
#         if curve_type == "SRVF":
#             energy_dot = True
#
#         original, boolean_mask, curve_hierarchy = \
#             shapedist.build_hierarchy_2D.hierarchical_curve_discretization(np.array([p, q]),
#                                                                            t1=t1, t2=t2,
#                                                                            init_coarsening_tol=hierarchy_tol,
#                                                                            n_levels=n_levels, max_iter=max_iter,
#                                                                            adaptive=adaptive,
#                                                                            interpolation_method=interpolation_method,
#                                                                            curve_type=curve_type)
#
#         t_orig = original[0]
#         b1_orig = original[1]
#         b2_orig = original[2]
#         if t_orig[boolean_mask[1]].size > 500:
#             warnings.warn("Algorithm will run slowly because curves are not coarsened enough."
#                           " A larger hierarchy tolerance is recommended.", RuntimeWarning)
#         tg, gammay, energy = shapedist.elastic_linear_hierarchy.find_gamma(t_orig, b1_orig, b2_orig, boolean_mask,
#                                                                                    energy_dot, gamma_tol)
#         if curve_type == "SRVF" and adaptive:
#             new_b2 = np.zeros((tg.size, 2))
#             new_b2[:, 0] = np.interp(gammay, tg, b2_orig[boolean_mask[1], 0])
#             new_b2[:, 1] = np.interp(gammay, tg, b2_orig[boolean_mask[1], 1])
#             new_b1 = b1_orig[boolean_mask[1]]
#             energy = shapedist.find_shape_distance_SRVF(tg, new_b1, new_b2)
#         elif curve_type == "SRVF" and not adaptive:
#             new_b2 = np.zeros((tg.size, 2))
#             new_b2[:, 0] = np.interp(gammay, tg, b2_orig[:, 0])
#             new_b2[:, 1] = np.interp(gammay, tg, b2_orig[:, 1])
#             new_b1 = b1_orig
#             energy = shapedist.find_shape_distance_SRVF(tg, new_b1, new_b2)
#
#         return tg, gammay, energy, original, boolean_mask

def elastic_matcher(p, q, t1=None, t2=None, curve_type="coord", gamma_tol=0.0001, adaptive=True, hierarchy_tol=1,
                    n_levels=3, max_iter=5, hierarchy_factor=2, energy_dot=False, interpolation_method='linear'):
    if len(p.shape) == 1 and len(q.shape) == 1:
        p = np.reshape(p, (-1, 1))
        q = np.reshape(q, (-1, 1))
    # Generate Hierarchy (coarsen curve in n dimensions) TODO
    [t, p, q], mask = shapedist.build_hierarchy.hierarchical_curve_discretization(p, q,
                                                                   t1=t1, t2=t2,
                                                                   init_coarsening_tol=hierarchy_tol,
                                                                   n_levels=n_levels, max_iter=max_iter,
                                                                   adaptive=adaptive,
                                                                   interpolation_method=interpolation_method,
                                                                   curve_type=curve_type)

    # Find gamma in N dimensions

    dim = p.shape[1]
    tg = t[mask[2]]
    gammay= np.zeros((t.shape[0], dim))

    energy = np.zeros(dim)
    for i in range(dim):
        temp, gammay[:, i], energy[i] = shapedist.elastic_linear_hierarchy.find_gamma(t, p[:,i], q[:,i], mask,
                                                                               energy_dot, gamma_tol)
    return p, q, tg, gammay, energy


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


@jit(float64(float64[:], float64[:], float64[:]), cache=True, nopython=True)
def inner_product(t, p, q):
    i = 0
    result = 0
    while i < p.size-1:
        result = result + (p[i] * q[i] + p[i+1] * q[i+1]) / 2 * (t[i+1] - t[i])
        i = i + 1
    return result


@jit(float64(float64[:], float64[:], float64[:]), cache=True, nopython=True)
def find_shape_distance(t, p, q):
    p_q = inner_product(t, p, q)
    p_p = inner_product(t, p, p)
    q_q = inner_product(t, q, q)
    temp = p_q / (p_p**0.5 * q_q ** 0.5)
    if temp > 1:
        temp = 1
    return np.arccos(temp) / pi


@jit(float64(float64[:], float64[:, :], float64[:, :]), cache=True, nopython=True)
def inner_product_2D(t, p, q):
    i = 0
    result = 0

    while i < p.shape[0]-1:
        val1 = p[i][0] * q[i][0] + p[i][1] * q[i][1]
        val2 = p[i+1][0] * q[i+1][0] + p[i+1][1] *q[i+1][1]
        result = result + (val1 + val2) / 2 * (t[i+1] - t[i])
        i = i + 1
    return result


@jit(float64(float64[:], float64[:, :], float64[:, :]), cache=True, nopython=True)
def find_shape_distance_SRVF(t, p, q):
    p_q = inner_product_2D(t, p, q)
    p_p = inner_product_2D(t, p, p)
    q_q = inner_product_2D(t, q, q)
    temp = p_q / (p_p**0.5 * q_q ** 0.5)
    if temp > 1:
        temp = 1
    return np.arccos(temp) / pi

