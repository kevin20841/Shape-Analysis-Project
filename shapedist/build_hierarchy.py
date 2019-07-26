import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import sys
np.set_printoptions(threshold=sys.maxsize)


def hierarchical_curve_discretization(p, q, t1, t2, init_coarsening_tol, uniform):

    # Curves should be an array of coordinates

    t, p, q = parametrize_curve_pair(p, q, t1, t2, 0.001)
    if uniform:
        boolean_mask = get_uniform_mask(p.shape[0])
    else:
        boolean_mask = get_adaptive_mask(p, q, t)
    return [t, p, q], boolean_mask


def get_adaptive_mask(p, q, t, init_coarsening_tol=None):
    t_new = t
    boolean_mask = np.zeros((2, t.shape[0]), dtype=np.bool)
    tol = init_coarsening_tol
    # get first adaptive mask
    # while t_new.size > 400:
    #     t_new, p, q = coarsen_curve(t_new, p, q, tol)
    #     tol = tol * 1.2
    t_new, temp1, temp2 = coarsen_curve_pair(t_new, p, q, 0.001)
    boolean_mask[1] = np.in1d(t, t_new)
    t_new = t
    t_new, temp1, temp2 = coarsen_curve_pair(t_new, p, q, 1)
    # print(t_new.shape, file=sys.stderr)
    # while t_new.size > 90:
    #     t_new, p, q = coarsen_curve(t_new, p, q, tol)
    #     tol = tol * 1.5
    # get second adaptive mask

    boolean_mask[0] = np.in1d(t, t_new)
    return boolean_mask


def get_uniform_mask(n):
    boolean_mask = []
    c = n
    ct = 1
    while c >= 64:
        mask = np.zeros(n, dtype=np.bool)
        for i in range(c):
            mask[i * ct] = True
        mask[-1] = True
        boolean_mask.append(mask)
        ct = ct * 2
        c = (c-1)//2

    boolean_mask[::-1] = boolean_mask
    boolean_mask = [boolean_mask[0], boolean_mask[2], boolean_mask[-1]]
    boolean_mask = np.array(boolean_mask, dtype=np.bool)

    return boolean_mask
    # boolean_mask = np.zeros((3, n), dtype=np.bool)
    # level_numbers = [60, 200, n]
    # for j in range(3):
    #     for i in range(level_numbers[j]):
    #         step_size = np.int(np.ceil(n / level_numbers[j]))
    #         if i * step_size < n:
    #             boolean_mask[j][i * step_size] = True
    #     boolean_mask[j][-1] = True
    # return boolean_mask


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
    node_markers_1[1:N-1] = np.logical_and(element_markers_1[0:N-2], element_markers_1[1:N-1])
    k = 0
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
    e = max_k * element_sizes**2

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
    K = -2 * K / np.sqrt(bottom_sqr)

    K = np.append(K, K[0])

    return K


def coarsen_curve(b, tol=2e-3, maxiter=5):
    i = 0

    while i < maxiter:
        element_errors_1 = geometric_discretization_error(b)
        markers = mark_nodes_for_coarsening(element_errors_1,  tol)
        if not(np.any(markers)):
            break

        b = b[np.logical_not(markers), :]
        # t = t[np.logical_not(markers)]
        i = i + 1
    # return t, b
    return b


def coarsen_curve_pair(t, b1, b2, tol=2e-3, maxiter=5):
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


def parametrize_curve_pair(p, q, t1, t2, t_spacing_tol):
    t = np.union1d(t1, t2)
    # N = t.shape[0]
    # remove = np.zeros(N, dtype=bool)
    # remove[1:N - 1] = (t[1:N - 1] - t[0:N - 2]) < t_spacing_tol
    # t = t[np.logical_not(remove)]

    dim = p.shape[1]
    ip = np.zeros((t.shape[0], dim))
    iq = np.zeros((t.shape[0], dim))
    for i in range(p.shape[1]):
        ip[:, i] = np.interp(t, t1, p[:, i])
        iq[:, i] = np.interp(t, t2, q[:, i])
    return t, ip, iq

