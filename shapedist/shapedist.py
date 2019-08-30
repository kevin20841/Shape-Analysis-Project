import shapedist
import numpy as np
from numba import jit, float64
from math import pi
from inspect import signature
from scipy.integrate import trapz
from scipy.linalg import svd
import scipy.interpolate
import sys
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

@jit(nopython=True)
def arclen_fct_values(b):
    """
    Returns an arclength parametrization of a curve in 2D
    Parameters
    ----------
    b : array of floats
        The curve
    Returns
    -------
    array of float
        The arclength parametrization
    """
    N = b.shape[0]
    d = np.zeros(N)
    d[1:N] = np.sum((b[1:N, :] - b[0:N-1, :])**2, 1)**0.5

    cumsum_d = np.cumsum(d)
    return cumsum_d / cumsum_d[N-1]

def normalize(p):
    """
    Normalizes a 2D curve by center of mass, and also makes its arclength 1
    Parameters
    ----------
    p : array of floats
        The curve to be noramlized

    Returns
    -------
    array of floats
        The normalized curve
    """
    N = p.shape[0]
    p = (p - shapedist.shape_representations.calculate_com(p))
    arclen_1 = np.sum((p[1:N, :] - p[0:N - 1, :]) ** 2, 1) ** 0.5
    arclen_1 = np.sum(arclen_1)
    p = p / arclen_1
    return p


def find_shapedist(p, q, dr='m', neigh = 5, shape_rep=shapedist.coords, distfunc=None, t1=None, t2=None,
                   tol=2e-3, energy_dot=False, strip_height=8):
    """

    Parameters
    ----------
    p
    q
    dr
    neigh
    shape_rep
    distfunc
    t1
    t2
    tol
    energy_dot
    strip_height

    Returns
    -------

    """
    uniform = False
    if 'u' in dr.lower():
        uniform = True
    coarsen = False
    if 'c' in dr.lower():
        coarsen = True
    if 't' in dr.lower():
        p = p.T
        q = q.T
    numparams = 1
    if len(p.shape) == 2 and p.shape[1] == 2:
        # p = shapedist.normalize(p)
        # q = shapedist.normalize(q)

        if t1 is None and t2 is None:
            t1 = arclen_fct_values(p)
            t2 = arclen_fct_values(q)
            uniform = False
        numparams = len(signature(shape_rep).parameters)

    if len(p.shape) == 1 and len(q.shape) == 1:
        p = np.reshape(p, (-1, 1))
        q = np.reshape(q, (-1, 1))
    c = False
    if shape_rep is shapedist.curvature:
        c = True
    [t, p, q], mask = shapedist.build_hierarchy.hierarchical_curve_discretization(p, q,
                                                                                  t1, t2,
                                                                                  coarsen, tol=tol, curvature=c)
    if numparams == 2:
        p, s1 = shape_rep(p, t)
        q, s2 = shape_rep(q, t)
    else:
        p, s1 = shape_rep(p)
        q, s2 = shape_rep(q)

    if shape_rep is shapedist.srvf:
        energy_dot = True
        distfunc = shapedist.calculate_shape_distance_SRVF
    if shape_rep is shapedist.tangent:
        distfunc = shapedist.calculate_shape_distance_tangent
    # Find gamma in N dimensions
    if len(p.shape) == 1 or p.shape[1] == 1:
        p = np.reshape(p, (-1))
        q = np.reshape(q, (-1))

    if len(p.shape) == 2:
        dim = p.shape[1]
    else:
        dim = 1
    if "2" in dr.lower():

        tg, gammay, sdist = shapedist.elastic_n_2.find_gamma(t, p, q, neigh, neigh, energy_dot, uniform, dim)
    else:
        tg, gammay, sdist = shapedist.elastic_linear_multilevel.find_gamma(t, p, q, mask, energy_dot, uniform, dim, neigh, strip_height)

    if distfunc is not None:
        sdist = distfunc(p, q, tg, gammay)
    if 'd' in dr.lower():
        if 't' in dr.lower():
            p = p.T
            q = q.T
        return sdist, p[mask[-1]], q[mask[-1]], tg, gammay
    else:
        return sdist



@jit(float64(float64[:], float64[:], float64[:]), cache=True, nopython=True)
def find_error(tg, gammar, gammat):
    """
    Calculates the difference between two gamma curves.
    Parameters
    ----------
    tg
    gammar
    gammat

    Returns
    -------

    """
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
    """
    Finds the inner product for SRVF and Tangent Shape Distance

    Parameters
    ----------
    t
    p
    q

    Returns
    -------

    """
    i = 0
    result = 0
    while i < p.size - 1:
        result = result + (p[i] * q[i] + p[i + 1] * q[i + 1]) / 2 * (t[i + 1] - t[i])
        i = i + 1
    return result


@jit(float64(float64[:], float64[:, :], float64[:, :]), cache=True, nopython=True)
def inner_product_2D(t, p, q):
    """
    Finds the inner product for SRVF and Tangent Shape Distance

    Parameters
    ----------
    t
    p
    q

    Returns
    -------

    """
    i = 0
    result = 0

    while i < p.shape[0] - 1:
        val1 = p[i][0] * q[i][0] + p[i][1] * q[i][1]
        val2 = p[i + 1][0] * q[i + 1][0] + p[i + 1][1] * q[i + 1][1]
        result = result + (val1 + val2) / 2 * (t[i + 1] - t[i])
        i = i + 1
    return result

def calculate_shape_distance_tangent(p, q, t, gamma):
    q_reparam = np.zeros(q.shape)
    for i in range(q.shape[1]):
        func = scipy.interpolate.CubicSpline(t, q[:, i])
        q_reparam[:, i] = func(gamma)
    q = q_reparam
    p_q = inner_product_2D(t, p, q)
    p_p = inner_product_2D(t, p, p)
    q_q = inner_product_2D(t, q, q)
    temp = p_q / (p_p ** 0.5 * q_q ** 0.5)
    if temp > 1:
        temp = 1
    return np.arccos(temp) / np.pi


def calculate_shape_distance_SRVF(p, q, t, gamma):
    gammad = np.sqrt(np.gradient(gamma, t))
    q_reparam = np.zeros(q.shape)
    for i in range(q.shape[1]):
        func = scipy.interpolate.CubicSpline(t, q[:, i])
        q_reparam[:, i] = func(gamma)
        q_reparam[:, i] = np.multiply(q_reparam[:, i], gammad)
    q = q_reparam
    p_q = inner_product_2D(t, p, q)
    p_p = inner_product_2D(t, p, p)
    q_q = inner_product_2D(t, q, q)
    temp = p_q / (p_p ** 0.5 * q_q ** 0.5)
    if temp > 1:
        temp = 1
    return np.arccos(temp) / pi

