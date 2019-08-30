import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import sys
from scipy.interpolate import CubicSpline
np.set_printoptions(threshold=sys.maxsize)


def hierarchical_curve_discretization(p, q, t1, t2, coarsen,  tol = 2e-3, curvature = False):
    # Curves should be an array of coordinates
    if coarsen:
        t = combine_t(t1, t2)
        t_p, p = coarsen_curve(t1, p, tol)
        mask = np.in1d(t, t_p)
        t_q, q = coarsen_curve(t2, q, tol)
        mask = np.logical_or(mask, np.in1d(t, t_q))
        mask[0] = True
        mask[-1] = True
        t = t[mask]
        p, q = parametrize_curve_pair(p, q, t, t_p, t_q, curvature)

        return [t, p, q], get_uniform_mask(t.shape[0])
    else:
        m = max(t1.shape[0], t2.shape[0])
        t = np.linspace(0., 1., m)
        p, q = parametrize_curve_pair(p, q, t, t1, t2, curvature)
        return [t, p, q], get_uniform_mask(t.shape[0])


def combine_t(t1, t2, t_spacing_tol=1e-4):
    t_spacing_tol = 1e-4
    t = np.union1d(t1, t2)
    N = t.shape[0]
    remove = np.zeros(N, dtype=bool)
    remove[1:N - 1] = (t[1:N - 1] - t[0:N - 2]) < t_spacing_tol
    k = 1
    while k < remove.shape[0]-2:
        if remove[k] and remove[k+1]:
            remove[k+1] = False
        k = k + 1

    t = t[np.logical_not(remove)]
    if t.shape[0] < 40:
        return np.union1d(t1, t2)
    return t


def get_uniform_mask(n):
    boolean_mask = []
    c = n
    ct = 1
    while c > 45:
        mask = np.zeros(n, dtype=np.bool)
        for i in range(c):
            mask[i * ct] = True
        mask[-1] = True
        boolean_mask.append(mask)
        ct = ct * 2
        c = (c)//2

    boolean_mask= boolean_mask[::-1]
    boolean_mask = np.array(boolean_mask, dtype=np.bool)

    return boolean_mask


def mark_nodes_for_coarsening_pair(element_errors_1, element_errors_2, t, tol, t_spacing_tol=0.0001):
    N = element_errors_1.size + 1
    element_markers_1 = element_errors_1 < tol
    node_markers_1 = np.ones(N) < 0
    node_markers_1[1:N-1] = np.logical_and(element_markers_1[0:N-2], element_markers_1[1:N-1])
    k = 0
    while k < N-1:
        if node_markers_1[k] and node_markers_1[k+1]:
            node_markers_1[k+1] = False
        k = k + 1

    N = element_errors_2.size + 1
    element_markers_2 = element_errors_2 < tol
    node_markers_2 = np.ones(N) < 0
    node_markers_2[1:N-1] = np.logical_and(element_markers_2[0:N-2], element_markers_2[1:N-1])
    k = 0
    while k < N-1:
        if node_markers_2[k] and node_markers_2[k+1]:
            node_markers_2[k+1] = False
        k = k + 1
    return np.logical_and(node_markers_1, node_markers_2)


def mark_nodes_for_coarsening(element_errors_1, tol):
    N = element_errors_1.size + 1
    element_markers_1 = element_errors_1 < tol
    node_markers_1 = np.ones(N) < 0
    node_markers_1[1:N-2] = np.logical_and(element_markers_1[0:N-3], element_markers_1[1:N-2])
    k = 1
    while k < N-1:
        if node_markers_1[k] and node_markers_1[k+1]:
            node_markers_1[k+1] = False
        k = k + 1
    return node_markers_1


def geometric_discretization_error(b):
    T = b[1:b.shape[0]] - b[0:b.shape[0]-1]
    element_sizes = np.sqrt(np.sum(T**2, 1))
    K = np.abs(curvature(b))
    max_k = np.maximum(K[0:K.size-1], K[1:K.size])
    e = np.multiply(max_k, element_sizes**2)

    return e

def curvature(p):
    n = p.shape[0]
    x = p[:, 0]
    y = p[:, 1]
    d = (x[1:n] - x[0:n - 1]) ** 2 + (y[1:n] - y[0: n - 1]) ** 2
    y1 = y[1:n] - y[0: n - 1]
    y2 = np.zeros(n-1)
    y2[1: n - 1] = y[2: n] - y[0: n - 2]
    y2[0] = y[1] - y[n - 2]

    d2 = np.zeros(n - 1)
    d2[1:n-1] = y2[1:n-1] ** 2 + (x[2:n] - x[0:n-2]) ** 2
    d2[0] = y2[0] ** 2 + (x[1] - x[n - 1]) ** 2
    bottom_sqr = np.zeros(n - 1)
    bottom_sqr[1: n-1] = d[1:n-1] * d[1:n-1] * d[0:n-2]
    bottom_sqr[0] = d[0] * d2[0] * d[n-2]

    K = np.zeros(n-1)
    K[1:n-1] = x[0:n-2] * y1[1:n-1] - x[1: n-1] * y2[1:n-1] + x[2:n] * y1[0:n-2]
    K[0] = x[n-2] * y1[0] - x[0] * y2[0] + x[1] * y1[n-3]
    bottom_sqr[bottom_sqr< 1e-14] = 1e-14
    K = -2 * K / np.sqrt(bottom_sqr)

    K = np.append(K, K[0])

    return K


def coarsen_curve(t, b, tol=2e-3, maxiter=15):
    i = 0
    while i < maxiter:
        element_errors_1 = geometric_discretization_error(b)
        markers = mark_nodes_for_coarsening(element_errors_1,  tol)
        if not(np.any(markers)):
            break

        b = b[np.logical_not(markers), :]
        t = t[np.logical_not(markers)]
        i = i + 1
    return t, b



def coarsen_curve_pair(t, b1, b2, tol=2e-3, maxiter=7):
    i = 0

    while i < maxiter:
        element_errors_1 = geometric_discretization_error(b1)
        element_errors_2 = geometric_discretization_error(b2)
        markers = mark_nodes_for_coarsening_pair(element_errors_1, element_errors_2, t, tol)
        if not(np.any(markers)):
            break

        b1 = b1[np.logical_not(markers), :]
        b2 = b2[np.logical_not(markers), :]
        t = t[np.logical_not(markers)]
        i = i + 1
    return t, b1, b2


def parametrize_curve_pair(p, q, t, t1, t2,curvature):
    dim = p.shape[1]
    ip = np.zeros((t.shape[0], dim))
    iq = np.zeros((t.shape[0], dim))
    curvature = True
    if not curvature:
        for i in range(p.shape[1]):
            ip[:, i] = np.interp(t, t1, p[:, i])
            iq[:, i] = np.interp(t, t2, q[:, i])
        return ip, iq
    else:
        for d in range(p.shape[1]):
            # N = t1.shape[0]
            # print(t1[1:N] == t1[0:N-1])
            inter_p = CubicSpline(t1, p[:, d])
            ip[:, d] = inter_p(t)

            # print(t2)
            inter_q = CubicSpline(t2, q[:, d])
            iq[:, d] = inter_q(t)
        return ip, iq