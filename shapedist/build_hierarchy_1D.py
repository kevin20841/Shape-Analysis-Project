import numpy as np


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
    element_sizes = np.sqrt(np.sum(T**2))
    K = np.abs(np.gradient(np.gradient(b)))
    max_k = np.maximum(K[0:K.size-1], K[1:K.size])
    e = max_k * element_sizes ** 2
    return e


def coarsen_curve(t, b1, b2, tol=2e-7, maxiter=5):
    i = 0
    while i < maxiter:
        element_errors_1 = geometric_discretization_error(b1)
        element_errors_2 = geometric_discretization_error(b2)
        markers = mark_nodes_for_coarsening(element_errors_1, element_errors_2, tol)
        if not(np.any(markers)):
            break

        b1 = b1[np.logical_not(markers)]
        b2 = b2[np.logical_not(markers)]
        t = t[np.logical_not(markers)]
        i = i + 1
    return t, b1, b2


def hierarchical_curve_discretization(curves, init_coarsening_tol=2e-3,
                                      n_levels=3, max_iter=5, adaptive=True, hierarchy_factor = 2,
                                      interpolation_method="linear"):

    single_curve = len(curves) == 1
    if single_curve:
        t = curves[0][0]
        b = curves[0][1]
        b = (b - np.min(b)) / np.max(b - np.min(b))
        t = (t - np.min(t)) / np.max(t - np.min(t))
    else:
        t1 = curves[0][0]
        t2 = curves[1][0]
        b1 = curves[0][1]
        b2 = curves[1][1]

        t1 = (t1 - np.min(t1)) / np.max(t1 - np.min(t1))
        t2 = (t1 - np.min(t2)) / np.max(t2 - np.min(t2))
        b1 = (b1 - np.min(b1)) / np.max(b1 - np.min(b1))
        b2 = (b2 - np.min(b2)) / np.max(b2 - np.min(b2))

    hierarchy = []
    if single_curve:
        tol = init_coarsening_tol
        count = 0

        t_prev = t.size * 2
        hierarchy.append([t, b])
        original = [t, b]
        while count < n_levels:
            t, b_coarse, b_coarse = coarsen_curve(t, b, b, tol, max_iter)
            if adaptive:
                if t.size == t_prev:
                    break
                elif t.size <= 90:
                    hierarchy.append([t, b_coarse])
                    break
                elif t.size * 2 < t_prev:
                    hierarchy.append([t, b_coarse])
                    t_prev = t.size

            else:
                if t.size == t_prev:
                    raise RuntimeError("Curves cannot be coarsened more than " +
                                       str(count) + " levels with given parameters (given " + str(n_levels)
                                       + " levels).")
                elif t.size * 2 < t_prev:
                    hierarchy.append([t, b_coarse])
                    count = count + 1

            b = b_coarse
            tol = 2 * tol
        if len(hierarchy) == 2:
            hierarchy.insert(0, original)
        n_levels = len(hierarchy)

        N = hierarchy[0][0].size
        boolean_mask = np.zeros([n_levels, N]) < 1

        t = hierarchy[0][0]
        for i in range(n_levels - 1):
            t2 = hierarchy[i+1][0]
            boolean_mask[i+1] = np.in1d(t, t2)
        for i in range(n_levels):
            boolean_mask[i][-1] = True
            boolean_mask[i][0] = True

        return original, boolean_mask[::-1][:-1], hierarchy[::-1][:-1]
    else:

        tol = init_coarsening_tol
        t_spacing_tol = 0.0001

        b1_combined, b2_combined, t_new = parametrize_curve_pair(b1, b2, t1, t2)
        N = t_new.size
        remove = np.zeros(N, dtype=bool)
        remove[1:N - 1] = (t_new[1:N - 1] - t_new[0:N - 2]) < t_spacing_tol
        t_new = t_new[np.logical_not(remove)]
        b1_combined = b1_combined[np.logical_not(remove)]
        b2_combined = b2_combined[np.logical_not(remove)]

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
            b1_coarse = b1_coarse[np.logical_not(remove)]
            b2_coarse = b2_coarse[np.logical_not(remove)]
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
        n_levels = len(hierarchy)
        N = hierarchy[0][0].size
        boolean_mask = np.zeros([n_levels, N]) < 1

        t = hierarchy[0][0]
        for i in range(n_levels - 1):
            t2 = hierarchy[i+1][0]
            boolean_mask[i+1] = np.in1d(t, t2)
        for i in range(n_levels):
            boolean_mask[i][-1] = True
            boolean_mask[i][0] = True

        return original, boolean_mask[::-1][:-1], hierarchy[::-1][:-1]


def parametrize_curve_pair(b1_in, b2_in, t1, t2):
    t = np.union1d(t1, t2)
    b1 = np.interp(t, t1, b1_in)
    b2 = np.interp(t, t2, b2_in)
    return b1, b2, t
