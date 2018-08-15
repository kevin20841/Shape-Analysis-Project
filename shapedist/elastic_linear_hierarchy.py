"""
Implementation of linear algorithm to determine shape elasticity as described by Bernal et al.
"""
import numpy as np
from numba import jit, types, float64, int16, boolean, generated_jit
import shapedist.elastic_n_2
from math import floor, pi


@jit([float64(float64, float64[:], float64[:], int16, int16)], cache=True, nopython=True)
def interp(t, x, y, lower, upper):
    i = 0
    while lower < upper:
        i = lower + (upper - lower) // 2
        val = x[i]
        if t == val:
            break
        elif t > val:
            if lower == i:
                break
            lower = i
        elif t < val:
            upper = i

    if i == x.size-1:
        temp = y[i]
    else:
        temp = (t - x[i]) * (y[i + 1] - y[i]) / (x[i + 1] - x[i]) + y[i]
    return temp


@jit([float64(float64, float64[:], float64, int16, int16)], cache=True, nopython=True)
def interp_range_only(t, y, n, lower, upper):
    i = 0
    while lower < upper:
        i = lower + (upper - lower) // 2
        val = (1/n) * i
        if t == val:
            break
        elif t > val:
            if lower == i:
                break
            lower = i
        elif t < val:
            upper = i
    return (t - (1/n) * i) * (y[i + 1] - y[i]) / ((1/n) * (i + 1) - (1/n) * i) + y[i]


def integrate_1D(tp, tq, py, qy, k, i, l, j, gamma_interval):
    e = 0
    a = k
    while a < i:
        gammak_1 = gamma_interval * l + (tp[a] - tp[k]) * \
                                        (gamma_interval * j - gamma_interval * l) / (tp[i] - tp[k])
        gammak_2 = gamma_interval * l + (tp[a+1] - tp[k]) * (gamma_interval * j - gamma_interval * l) \
                                        / (tp[i] - tp[k])
        e = e + (0.5 * (py[a] - interp(gammak_1, tq, qy, 0, tq.size)) ** 2
                         + 0.5 * (py[(a+1)] - interp(gammak_2, tq, qy, 0, tq.size)) ** 2) * \
                (tp[a+1] - tp[a]) * 0.5
        a = a + 1
    return e


def integrate_2D(tp, tq, py, qy, k, i, l, j, gamma_interval):
    e = 0
    a = k
    while a < i:
        gammak_1 = gamma_interval * l + (tp[a] - tp[k]) * (gamma_interval * j - gamma_interval * l) / (tp[i] - tp[k])
        gammak_2 = gamma_interval * l + (tp[a+1] - tp[k]) * (gamma_interval * j - gamma_interval * l) / (tp[i] - tp[k])

        qx_value_1 = interp(gammak_1, tq, qy[:, 0], 0, tq.size)
        qy_value_1 = interp(gammak_1, tq, qy[:, 1], 0, tq.size)

        qx_value_2 = interp(gammak_2, tq, qy[:, 0], 0, tq.size)
        qy_value_2 = interp(gammak_2, tq, qy[:, 1], 0, tq.size)

        val1 = 0.5 * (py[a][0] ** 2 + py[a][1] ** 2
                      - 2*(py[a][0] * qx_value_1 + py[a][1] * qy_value_1)
                      + qx_value_1**2 + qy_value_1**2)
        val2 = 0.5 * (py[a+1][0] ** 2 + py[a+1][1] ** 2
                      - 2 * (py[a+1][0] * qx_value_2 + py[a+1][1] * qy_value_2)
                      + qx_value_2 ** 2 + qy_value_2 ** 2)
        e = e + (val1 + val2) * (tp[a+1] - tp[a]) * 0.5
        a = a + 1
    return e


@generated_jit(nopython=True)
def integrate(tp, tq, py, qy, k, i, l, j, gamma_interval):
    if py.ndim == 1:
        return integrate_1D
    elif py.ndim == 2:
        return integrate_2D

# @jit([types.Tuple((float64[:], float64[:], float64))(float64[:], float64[:], float64[:], boolean[:, :], int16, int16)],
#      cache=True, nopython=True)


@jit(cache=True, nopython=True)
def find_gamma(t, p, q, parametrization_array, neighborhood, strip_height):
    # Initialize variables
    max_iteration = parametrization_array.shape[0]
    tp = t
    tq = t
    py = p
    qy = q
    current_iteration = 0
    n = tp.size
    initial_size = 0
    i = 0
    while i < parametrization_array[0].shape[0]:
        if parametrization_array[0][i]:
            initial_size = initial_size + 1
        i = i + 1

    index_array = np.linspace(0, n-1, n)
    path = np.zeros(n, dtype=np.float64)
    # Initial N^2 path
    tg = np.linspace(0., 1., n).astype(np.float64)
    g = np.linspace(0., 1., initial_size).astype(np.float64)

    gamma_domain, gamma_range, energy = shapedist.elastic_n_2.find_gamma(
        tp[parametrization_array[0]][:initial_size], py[parametrization_array[0]][:initial_size],
        qy[parametrization_array[0]][:initial_size], tg[parametrization_array[0]][:initial_size],
        g, 4, 4)
    i = 0
    while i < gamma_range.size:
        path[i] = gamma_range[i]
        i = i + 1
    current_iteration = current_iteration + 1

    # Linear Iteration
    i = 0
    min_energy_values = np.zeros((n, n), dtype=np.float64)
    path_nodes = np.zeros((n, n, 2), dtype=np.int16)
    previous_n = gamma_range.size
    previous_parametrization_size = initial_size

    while current_iteration < max_iteration:
        i = 0
        parametrization_size = 0
        while i < parametrization_array.shape[1]:
            if parametrization_array[current_iteration][i]:
                parametrization_size = parametrization_size + 1
            i = i + 1
        m = parametrization_size
        n = parametrization_size
        temp_domain_gamma = tg[parametrization_array[current_iteration]][:n]
        prev_temp_domain_gamma = tg[parametrization_array[current_iteration-1]][:previous_n]
        tp_temp = tp[parametrization_array[current_iteration]][:n]
        py_temp = py[parametrization_array[current_iteration]][:n]
        tq_temp = tq[parametrization_array[current_iteration]][:n]
        qy_temp = qy[parametrization_array[current_iteration]][:n]
        min_energy_values = np.zeros((n, n), dtype=np.float64)
        path_nodes = np.zeros((n, n, 2), dtype=np.int16)
        gamma_interval = 1 / (m-1)
        min_energy_values[0][0] = integrate(tp_temp, tq_temp, py_temp, qy_temp, 0, 1, 0, 1, gamma_interval)

        path_nodes[1][1][0] = 0
        path_nodes[1][1][1] = 0
        i, j, k, l = 1, 1, 1, 1
        val = 0
        val2 = 0

        while i < n-1:
            val = interp(tp_temp[i], prev_temp_domain_gamma,
                         path, 0, previous_n)
            j = floor(val / gamma_interval) - strip_height
            if j <= 0:
                j = 1
            while j < m-1 and j * gamma_interval < val + strip_height * gamma_interval:
                min_energy_values[i][j] = integrate(tp_temp, tq_temp, py_temp, qy_temp, 0, i, 0, j,
                                                    gamma_interval)

                k = i - neighborhood
                if k <= 0:
                    k = 1
                minimum = min_energy_values[i][j]
                while k < i:
                    # val2 = interp_range_only(tp[rough_path[k]], path, previous_n, 0,
                    #                          previous_n)
                    val2 = interp(tp_temp[k],
                                  prev_temp_domain_gamma,
                                  path, 0, previous_n)
                    l = j - neighborhood
                    if l <= floor(val2 / gamma_interval) - strip_height:
                        l = floor(val2 / gamma_interval) - strip_height
                    if l <= 0:
                        l = 1
                    while l < j and l * gamma_interval < val2 + strip_height * gamma_interval:
                        e = min_energy_values[k, l] + integrate(tp_temp, tq_temp, py_temp, qy_temp,
                                                                    k, i, l, j,
                                                                    gamma_interval)
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
        i = n-1
        j = m-1
        min_energy_values[i][j] = integrate(tp_temp, tq_temp, py_temp, qy_temp, 0, i, 0, j, gamma_interval)
        k = i - neighborhood
        if k <= 0:
            k = 1
        minimum = min_energy_values[i][j]
        while k < i:
            val2 = interp(tp_temp[k],
                          prev_temp_domain_gamma,
                                  path, 0, previous_n)
            l = j - neighborhood
            if l <= floor(val2 / gamma_interval) - strip_height:
                l = floor(val2 / gamma_interval) - strip_height
            if l <= 0:
                l = 1
            while l < j and l * gamma_interval < val2 + strip_height * gamma_interval:
                e = min_energy_values[k, l] + integrate(tp_temp, tq_temp, py_temp, qy_temp,
                                                        k, i, l, j,
                                                        gamma_interval)

                if e < minimum:
                    minimum = e
                    path_nodes[i][j][0] = k
                    path_nodes[i][j][1] = l
                l = l + 1
            k = k + 1

        min_energy_values[i][j] = minimum

        # !! Interpolate
        path_indices = np.zeros((n, 2), dtype=np.int16)
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
                                                            (temp_domain_gamma[previousIndex_domain] - temp_domain_gamma[path_indices[i][0]])
                while j < previousIndex_domain - path_indices[i][0]:
                    path[previousIndex_domain - j] = previous - (temp_domain_gamma[previousIndex_domain] -
                                                                 temp_domain_gamma[previousIndex_domain - j]) * val


                    j = j + 1
            previousIndex_domain = path_indices[i][0]
            previousIndex_gamma = path_indices[i][1]
            previous = gamma_interval * path_indices[i][1]
            i = i + 1
        previous_n = n
        current_iteration = current_iteration + 1

    return tg[parametrization_array[current_iteration-1]][:n], path[:n], min_energy_values[n-1][m-1]


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
