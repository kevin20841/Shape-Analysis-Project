import shapedist
import numpy as np
from numba import jit, float64
from math import pi


def find_shapedist(p, q, dr='', shape_rep=shapedist.coords, distfunc=None, t1=None, t2=None,
                   init_coarsening_tol=2e-6, energy_dot=False):

    if len(p.shape) == 1 and len(q.shape) == 1:
        p = np.reshape(p, (-1, 1))
        q = np.reshape(q, (-1, 1))
    # Generate Hierarchy (coarsen curve in n dimensions) TODO
    uniform = False
    if 'u' in dr.lower():
        uniform = True
    # Normalize curve to center of mass if two dimensions
    if p.shape[1] == 2:
        N = p.shape[0]
        arclen_1 = np.sum((p[1:N, :] - p[0:N - 1, :]) ** 2, 1) ** 0.5
        arclen_1 = np.sum(arclen_1)
        p = (p - shapedist.shape_representations.calculate_com(p)) / arclen_1

        N = q.shape[0]
        arclen_2 = np.sum(np.sum((q[1:N, :] - q[0:N - 1, :]) ** 2, 1) ** 0.5)
        arclen_2 = np.sum(arclen_2)
        q = (q - shapedist.shape_representations.calculate_com(q)) / arclen_2
        p = shape_rep(p)
        q = shape_rep(q)

    [t, p, q], mask = shapedist.build_hierarchy.hierarchical_curve_discretization(p, q,
                                                                                  t1, t2,
                                                                                  init_coarsening_tol,
                                                                                  uniform)
    # Find gamma in N dimensions
    dim = p.shape[1]
    tg = t[mask[2]]
    gammay = np.zeros((t.shape[0], dim))

    energy = np.zeros(dim)
    for i in range(dim):
        temp, gammay[:, i], energy[i] = shapedist.elastic_linear_hierarchy.find_gamma(t, p[:, i], q[:, i], mask,
                                                                                      energy_dot)
    if dim ==1:
        energy = energy[0]
    if gammay.shape[1] == 1:
        gammay = np.reshape(gammay, (-1))
        p = np.reshape(p, (-1))
        p = np.reshape(p, (-1))
    if distfunc is not None:
        energy = distfunc(energy)

    if 'd' in dr.lower():
        return energy, p, q, tg, gammay
    else:
        return energy


@jit(float64(float64[:], float64[:], float64[:]), cache=True, nopython=True)
def find_error(tg, gammar, gammat):
    n = tg.size
    error = 1 / 2 * (tg[1] - tg[0]) * (gammar[1] - gammat[1]) ** 2 + 1 / 2 * (tg[n - 1] - tg[n - 2]) * (
                gammar[n - 1] - gammat[n - 1]) ** 2
    k = 2
    if n != gammar.size or n != gammat.size:
        raise IndexError
    while k < n - 1:
        error = error + 1 / 2 * (gammar[k] - gammat[k]) ** 2 * (tg[k] - tg[k - 1]) ** 2
        k = k + 1
    error = error ** (1 / 2)
    return error


@jit(float64(float64[:], float64[:], float64[:]), cache=True, nopython=True)
def inner_product(t, p, q):
    i = 0
    result = 0
    while i < p.size - 1:
        result = result + (p[i] * q[i] + p[i + 1] * q[i + 1]) / 2 * (t[i + 1] - t[i])
        i = i + 1
    return result


@jit(float64(float64[:], float64[:], float64[:]), cache=True, nopython=True)
def find_shape_distance(t, p, q):
    p_q = inner_product(t, p, q)
    p_p = inner_product(t, p, p)
    q_q = inner_product(t, q, q)
    temp = p_q / (p_p ** 0.5 * q_q ** 0.5)
    if temp > 1:
        temp = 1
    return np.arccos(temp) / pi


@jit(float64(float64[:], float64[:, :], float64[:, :]), cache=True, nopython=True)
def inner_product_2D(t, p, q):
    i = 0
    result = 0

    while i < p.shape[0] - 1:
        val1 = p[i][0] * q[i][0] + p[i][1] * q[i][1]
        val2 = p[i + 1][0] * q[i + 1][0] + p[i + 1][1] * q[i + 1][1]
        result = result + (val1 + val2) / 2 * (t[i + 1] - t[i])
        i = i + 1
    return result


@jit(float64(float64[:], float64[:, :], float64[:, :]), cache=True, nopython=True)
def find_shape_distance_SRVF(t, p, q):
    p_q = inner_product_2D(t, p, q)
    p_p = inner_product_2D(t, p, p)
    q_q = inner_product_2D(t, q, q)
    temp = p_q / (p_p ** 0.5 * q_q ** 0.5)
    if temp > 1:
        temp = 1
    return np.arccos(temp) / pi
