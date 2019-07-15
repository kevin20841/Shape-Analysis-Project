import numpy as np
import shapedist.shape_representations


def arclen_fct_values(b):
    N = b[:, 0].size
    d = np.zeros(N)
    d[1:N] = np.sum((b[2:N, :] - b[1:N-1, :])**2)**0.5
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


def hierarchical_curve_discretization(curves, init_coarsening_tol=2e-3, n_levels=5,
                                      max_iter=5, adaptive=True, hierarchy_factor=2,
                                      interpolation_method="linear", curve_type="coord"):

    single_curve = len(curves) == 1
    if single_curve:
        b = curves[0]

    else:
        b1 = curves[0]
        b2 = curves[1]
        b1[:, 0] = (b1[:, 0] - np.min(b1[:, 0])) / np.max(b1[:, 0] - np.min(b1[:, 0]))
        b1[:, 1] = (b1[:, 1] - np.min(b1[:, 1])) / np.max(b1[:, 1] - np.min(b1[:, 1]))
        b2[:, 0] = (b2[:, 0] - np.min(b2[:, 0])) / np.max(b2[:, 0] - np.min(b2[:, 0]))
        b2[:, 1] = (b2[:, 1] - np.min(b2[:, 1])) / np.max(b2[:, 1] - np.min(b2[:, 1]))
        b2 = b1[0] - b2[0] + b2
    hierarchy = []
    if single_curve:
        tol = init_coarsening_tol
        for level in range(n_levels):
            b_coarse = coarsen_curve(b, tol, max_iter)
            t = arclen_fct_values(b_coarse)
            hierarchy[level] = [t, b_coarse]
            b = b_coarse
            tol = hierarchy_factor * tol

    else:

        tol = init_coarsening_tol
        t_spacing_tol = 0.0001
        t1 = arclen_fct_values(b1)
        t2 = arclen_fct_values(b2)
        b1_combined, b2_combined, t_new = parametrize_curve_pair(b1, b2, t1, t2,
                                                       interpolation_method=interpolation_method)
        t_new = t_new.transpose()
        b1_combined = b1_combined.transpose()
        b2_combined = b2_combined.transpose()
        N = t_new.size
        remove = np.zeros(N, dtype=bool)
        remove[1:N - 1] = (t_new[1:N - 1] - t_new[0:N - 2]) < t_spacing_tol
        t_new = t_new[np.logical_not(remove)]
        b1_combined = b1_combined[np.logical_not(remove), :]
        b2_combined = b2_combined[np.logical_not(remove), :]
        t_new, b1_combined, b2_combined = coarsen_curve(t_new, b1_combined, b2_combined, tol, max_iter)

        original = [t_new, b1_combined, b2_combined]
        hierarchy.append([t_new, b1_combined, b2_combined])
        count = 0
        t_prev = t_new.size * hierarchy_factor
        while count < n_levels:
            t_new, b1_coarse, b2_coarse = coarsen_curve(t_new, b1_combined, b2_combined, tol, max_iter)
            # b1 = b1_coarse
            # b2 = b2_coarse

            N = t_new.size
            remove = np.zeros(N, dtype=bool)
            remove[1:N-1] = (t_new[1:N-1] - t_new[0:N-2]) < t_spacing_tol
            t_new = t_new[np.logical_not(remove)]
            b1_coarse = b1_coarse[np.logical_not(remove), :]
            b2_coarse = b2_coarse[np.logical_not(remove), :]
            if adaptive:
                if t_new.size == t_prev:
                    break
                elif t_new.size <= 90:
                    hierarchy.append([t_new, b1_coarse, b2_coarse])
                    break
                elif t_new.size * hierarchy_factor < t_prev:
                    hierarchy.append([t_new, b1_coarse, b2_coarse])
                    t_prev = t_new.size

            else:
                if t_new.size == t_prev:
                    raise RuntimeError("Curves cannot be coarsened up to " + str(n_levels) +
                                       " levels with given parameters, but rather only up to " +
                                       str(count) + " levels.")
                elif t_new.size * hierarchy_factor < t_prev:
                    hierarchy.append([t_new, b1_coarse, b2_coarse])
                    count = count + 1
            b1_combined = b1_coarse
            b2_combined = b2_coarse
            tol = 2 * tol
        if len(hierarchy) == 2:
            hierarchy.insert(0, original)
        N = hierarchy[0][0].size
        n_levels = len(hierarchy)
        boolean_mask = np.zeros([n_levels, N]) < 1
        t = hierarchy[0][0]
        for i in range(n_levels - 1):
            t2 = hierarchy[i + 1][0]
            boolean_mask[i + 1] = np.in1d(t, t2)
        for i in range(n_levels):
            boolean_mask[i][-1] = True
            boolean_mask[i][0] = True
        temp = original[:]

        if curve_type == "curvature":
            original[1] = curvature(original[1])
            original[1] = (original[1] - np.min(original[1]))/np.max(original[1]-np.min(original[1]))
            original[2] = curvature(original[2])
            original[2] = (original[2] - np.min(original[2])) / np.max(original[2] - np.min(original[2]))
        elif curve_type == "normals":
            original[1] = shapedist.shape_representations.normals(original[1])
            original[2] = shapedist.shape_representations.normals(original[2])

        return original, boolean_mask[::-1][:-1], hierarchy[::-1][:-1]


def parametrize_curve_pair(b1_in, b2_in, t1, t2, interpolation_method='linear' ):
    if t1 is None and t2 is None:
        t1 = arclen_fct_values(b1_in)
        t2 = arclen_fct_values(b2_in)
    t = np.union1d(t1, t2)
    N = t.size
    dim = np.size(b1_in, 1)
    N = t.size
    b1 = np.zeros((dim, N))
    b2 = np.zeros((dim, N))
    for k in range(dim):
        b1[k, :] = np.interp(t, t1, b1_in[:, k])
        b2[k, :] = np.interp(t, t2, b2_in[:, k])
    return b1, b2, t