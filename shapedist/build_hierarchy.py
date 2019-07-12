import numpy as np
import shapedist.shape_distance_types


def hierarchical_curve_discretization(p, q, t1=None, t2=None, init_coarsening_tol=2e-6, n_levels=5,
                                      max_iter=3, adaptive=True,
                                      interpolation_method="linear", curve_type="coord"):

    # Curves should be an array of coordinates

    # Normalize curve to center of mass
    N = p.shape[0]
    # arclen_1 = np.sum((p[1:N, :] - p[0:N - 1, :]) ** 2, 1) ** 0.5
    # arclen_1 = np.sum(arclen_1)
    # p = (p - shapedist.shape_distance_types.calculate_com(p)) / arclen_1
    #
    # N = q.shape[0]
    # arclen_2 = np.sum(np.sum((q[1:N, :] - q[0:N - 1, :]) ** 2, 1) ** 0.5)
    # arclen_2 = np.sum(arclen_2)
    # q = (q - shapedist.shape_distance_types.calculate_com(q)) / arclen_2
    t, p, q = parametrize_curve_pair(p, q, t1, t2, init_coarsening_tol, interpolation_method=interpolation_method)
    # TODO Coarsen Curves (for now it just gets a boolean mask that doubles in size)

    boolean_mask = np.zeros((3, p.shape[0]), dtype=np.bool)
    N = p.shape[0]
    level_numbers = [60, 200, N]
    for j in range(3):
        for i in range(level_numbers[j]):
            step_size = np.int(np.ceil(p.shape[0] / level_numbers[j]))
            if i * step_size < p.shape[0]:
                boolean_mask[j][i * step_size] = True
        boolean_mask[j][-1] = True
    return [t, p, q], boolean_mask


def arclen_fct_values(b):
    N = b[:, 0].size
    d = np.zeros(N)
    d[1:N] = np.sum((b[1:N, :] - b[0:N-1, :])**2, 1)**0.5

    cumsum_d = np.cumsum(d)
    return cumsum_d / cumsum_d[N-1]


def mark_nodes_for_coarsening(element_errors_1, element_errors_2, tol):
    N = element_errors_1.size + 1
    element_markers_1 = element_errors_1 < tol
    node_markers_1 = np.ones(N) < 0
    node_markers_1[1:N-1] = np.logical_and(element_markers_1[0:N-2], element_markers_1[1:N-1])
    k = 1
    while k < N-1:
        if np.logical_and(node_markers_1[k], node_markers_1[k+1]):
            node_markers_1[k+1] = False
        k = k + 1
    N = element_errors_2.size + 1
    element_markers_2 = element_errors_2 < tol
    node_markers_2 = np.ones(N) < 0
    node_markers_2[1:N-1] = np.logical_and(element_markers_2[0:N-2], element_markers_2[1:N-1])
    k = 1
    while k < N-1:
        if np.logical_and(node_markers_2[k], node_markers_2[k+1]):
            node_markers_2[k+1] = False
        k = k + 1
    return np.logical_and(node_markers_1, node_markers_2)


def geometric_discretization_error(b):
    T = b[1:b.shape[0]] - b[0:b.shape[0]-1]
    element_sizes = np.sqrt(np.sum(T**2, 1))

    K = np.abs(curvature(b))
    max_k = np.maximum(K[0:K.size-1], K[1:K.size])
    e = max_k * element_sizes ** 2
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


def coarsen_curve(t, b1, b2, tol=2e-3, maxiter=5):
    i = 0

    while i < maxiter:
        element_errors_1 = geometric_discretization_error(b1)
        element_errors_2 = geometric_discretization_error(b2)
        markers = mark_nodes_for_coarsening(element_errors_1, element_errors_2, tol)
        if not(np.any(markers)):
            break
        b1 = b1[np.logical_not(markers), :]
        b2 = b2[np.logical_not(markers), :]
        t = t[np.logical_not(markers)]

        i = i + 1
    return t, b1, b2


def parametrize_curve_pair(p, q, t1, t2, t_spacing_tol, interpolation_method='linear'):
    if t1 is None and t2 is None:
        t1 = arclen_fct_values(p)
        t2 = arclen_fct_values(q)
    t = np.union1d(t1, t2)
    N = t.shape[0]
    # remove = np.zeros(N, dtype=bool)
    # remove[1:N - 1] = (t[1:N - 1] - t[0:N - 2]) < t_spacing_tol
    # t = t[np.logical_not(remove)]
    # print(remove)
    dim = p.shape[1]
    ip = np.zeros((t.shape[0], dim))
    iq = np.zeros((t.shape[0], dim))
    for i in range(p.shape[1]):
        ip[:, i] = np.interp(t, t1, p[:, i])
        iq[:, i] = np.interp(t, t2, q[:, i])
    return t, ip, iq

