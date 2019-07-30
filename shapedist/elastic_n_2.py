"""
Implementation of linear algorithm to determine shape elasticity as described by Bernal et al.
"""
import numpy as np
from numba import jit, types, float64, int16, generated_jit
from math import floor
from shapedist.comp import *


@jit(cache=False, nopython=True)
def find_gamma(t, p, q, height, width, energy_dot, u, dim):

    # Linear Iteration
    parametrization_size = p.shape[0]
    start = np.empty(dim, dtype=np.int64)
    val1 = np.empty(dim, dtype=np.float64)
    end = np.empty(dim, dtype=np.int64)
    val2 = np.empty(dim, dtype=np.float64)
    path = np.zeros(parametrization_size, dtype=np.float64)

    m = parametrization_size
    n = parametrization_size

    min_energy_values = np.full((n, n), np.inf, dtype=np.float64)
    path_nodes = np.zeros((n, n, 2), dtype=np.int64)
    gamma_interval = 1 / (m - 1)

    min_energy_values[0][0] = integrate(t, t, p, q, 0, 1, 0, 1, gamma_interval, energy_dot,
                                        dim, start, end, val1, val2, u)

    path_nodes[1][1][0] = 0
    path_nodes[1][1][1] = 0
    i, j, k, l = 1, 1, 1, 1

    while i < n - 1:

        j = 1
        while j < m-1:
            min_energy_values[i][j] = integrate(t, t, p, q, 0, i, 0, j,
                                                gamma_interval, energy_dot,
                                                dim, start, end, val1, val2, u)
            k = i - height
            if k <= 0:
                k = 1
            minimum = min_energy_values[i][j]
            while k < i:
                l = j - width
                if l <= 0:
                    l = 1
                while l < j:
                    e = min_energy_values[k, l] + integrate(t, t, p, q,
                                                            k, i, l, j,
                                                            gamma_interval, energy_dot,
                                                            dim, start, end, val1, val2, u)
                    if e < minimum:
                        minimum = e
                        path_nodes[i][j][0] = k
                        path_nodes[i][j][1] = l

                    l = l + 1
                k = k + 1
            min_energy_values[i][j] = minimum
            j = j + 1
        i = i + 1
    # !!
    i = n - 1
    j = m - 1
    min_energy_values[i][j] = integrate(t, t, p, q, 0, i, 0, j,
                                        gamma_interval, energy_dot,
                                        dim, start, end, val1, val2, u)

    k = i - height
    if k <= 0:
        k = 1
    minimum = min_energy_values[i][j]
    while k < i:
        l = j - width
        if l <= 0:
            l = 1

        while l < j:
            e = min_energy_values[k, l] + integrate(t, t, p, q,
                                                    k, i, l, j,
                                                    gamma_interval, energy_dot,
                                                    dim, start, end, val1, val2, u)
            if e < minimum:
                minimum = e
                path_nodes[i][j][0] = k
                path_nodes[i][j][1] = l
            l = l + 1
        k = k + 1

    min_energy_values[i][j] = minimum

    # !! Interpolate
    path_indices = np.zeros((n, 2), dtype=np.int64)
    path_indices[0][0] = n - 1
    path_indices[0][1] = m - 1

    i = 0
    while path_indices[i][0] != 0 or path_indices[i][1] != 0 and i + 1 < path.size:
        result = path_nodes[path_indices[i][0]][path_indices[i][1]]
        path_indices[i + 1][0] = result[0]
        path_indices[i + 1][1] = result[1]
        i = i + 1
    i = 0
    previous = 1
    previousIndex_domain = n - 1
    previousIndex_gamma = n - 1

    path[path_indices[0][0]] = gamma_interval * path_indices[0][1]
    while i < path_indices.size // 2 and previousIndex_domain != 0:
        path[path_indices[i][0]] = gamma_interval * path_indices[i][1]
        if previousIndex_domain - path_indices[i][0] > 1:
            j = 0
            val = (gamma_interval * (previousIndex_gamma - path_indices[i][1])) / \
                  (t[previousIndex_domain] - t[path_indices[i][0]])
            while j < previousIndex_domain - path_indices[i][0]:
                path[previousIndex_domain - j] = previous - (t[previousIndex_domain] -
                                                             t[previousIndex_domain - j]) * val

                j = j + 1
        previousIndex_domain = path_indices[i][0]
        previousIndex_gamma = path_indices[i][1]
        previous = gamma_interval * path_indices[i][1]
        i = i + 1
    return t, path, min_energy_values[n - 1][m - 1]
