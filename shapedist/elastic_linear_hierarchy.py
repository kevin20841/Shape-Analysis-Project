"""
Implementation of linear algorithm to determine shape elasticity as described by Bernal et al.
"""
import numpy as np
from numba import jit, float64, int16, generated_jit
import shapedist.elastic_n_2
from math import floor, pi
import matplotlib.pyplot as plt
from line_profiler import LineProfiler


@jit([float64(float64, float64[:], float64[:], int16, int16)], cache=True, nopython=True, fastmath=True)
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


def integrate_1D(tp, tq, py, qy, k, i, l, j, gamma_interval, energy_dot):
    e = 0
    a = k
    gammak_1 = gamma_interval * l + (tp[a] - tp[k]) * \
               (gamma_interval * j - gamma_interval * l) / (tp[i] - tp[k])
    gammak_2 = gamma_interval * l + (tp[a + 1] - tp[k]) * (gamma_interval * j - gamma_interval * l) \
               / (tp[i] - tp[k])
    e = e + (0.5 * (py[a] - interp(gammak_1, tq, qy, 0, tq.size)) ** 2
             + 0.5 * (py[(a + 1)] - interp(gammak_2, tq, qy, 0, tq.size)) ** 2) * \
        (tp[a + 1] - tp[a]) * 0.5
    a = a + 1
    while a < i:
        gammak_2 = gamma_interval * l + (tp[a+1] - tp[k]) * (gamma_interval * j - gamma_interval * l) \
                                        / (tp[i] - tp[k])
        e = e + (0.5 * (py[a] - interp(gammak_1, tq, qy, 0, tq.size)) ** 2
                         + 0.5 * (py[(a+1)] - interp(gammak_2, tq, qy, 0, tq.size)) ** 2) * \
                (tp[a+1] - tp[a]) * 0.5
        gammak_1 = gammak_2
        a = a + 1
    return e


def integrate_2D(tp, tq, py, qy, k, i, l, j, gamma_interval, energy_dot):
    e = 0
    a = k
    while a < i:
        gamma_derivative = (gamma_interval * j - gamma_interval * l) / (tp[i] - tp[k])
        gammak_1 = gamma_interval * l + (tp[a] - tp[k]) * gamma_derivative
        gammak_2 = gamma_interval * l + (tp[a+1] - tp[k]) * gamma_derivative

        qx_value_1 = interp(gammak_1, tq, qy[:, 0], 0, tq.size)
        qy_value_1 = interp(gammak_1, tq, qy[:, 1], 0, tq.size)

        qx_value_2 = interp(gammak_2, tq, qy[:, 0], 0, tq.size)
        qy_value_2 = interp(gammak_2, tq, qy[:, 1], 0, tq.size)
        if not energy_dot:
            val1 = 0.5 * (py[a][0] ** 2 + py[a][1] ** 2
                          - 2*(py[a][0] * qx_value_1 + py[a][1] * qy_value_1)
                          + qx_value_1**2 + qy_value_1**2)
            val2 = 0.5 * (py[a+1][0] ** 2 + py[a+1][1] ** 2
                          - 2 * (py[a+1][0] * qx_value_2 + py[a+1][1] * qy_value_2)
                          + qx_value_2 ** 2 + qy_value_2 ** 2)
        else:
            val1 = 0.5 * (py[a][0] ** 2 + py[a][1] ** 2
                          - 2 * gamma_derivative ** 0.5 * (py[a][0] * qx_value_1 + py[a][1] * qy_value_1)
                          + gamma_derivative * (qx_value_1 ** 2 + qy_value_1 ** 2))
            val2 = 0.5 * (py[a + 1][0] ** 2 + py[a + 1][1] ** 2
                          - 2 * gamma_derivative ** 0.5 * (py[a + 1][0] * qx_value_2 + py[a + 1][1] * qy_value_2)
                          + gamma_derivative * (qx_value_2 ** 2 + qy_value_2 ** 2))
        e = e + (val1 + val2) * (tp[a+1] - tp[a]) * 0.5
        a = a + 1
    return e


@generated_jit(nopython=True, cache=True, fastmath=True)
def integrate(tp, tq, py, qy, k, i, l, j, gamma_interval, energy_dot):
    if py.ndim == 1:
        return integrate_1D
    elif py.ndim == 2:
        return integrate_2D


@jit(nopython=True, cache=True, fastmath=True)
def running_mean(x, N):
    ret = np.zeros(x.size)
    for i in range(ret.size):
        back = i - N if (i - N) >=0 else 0
        front = i + N if (i + N) < ret.size else ret.size-1
        ret[i] = np.sum(x[back:front]) / (front - back)

    return ret


@jit(cache=True, nopython=True, fastmath=True)
def find_gamma(t, p, q, parametrization_array, energy_dot, gamma_tol):
    # Initialize variables
    tp = t
    tq = t
    py = p
    qy = q
    n_2_precision = 6

    initial_size = 0
    i = 0
    while i < parametrization_array[0].shape[0]:
        if parametrization_array[0][i]:
            initial_size = initial_size + 1
        i = i + 1

    previous_n = initial_size
    parametrization_size = 0
    i = 0
    while i < parametrization_array.shape[1]:
        if parametrization_array[1][i]:
            parametrization_size = parametrization_size + 1
        i = i + 1
    n = tp.size
    path = np.zeros(parametrization_size, dtype=np.float64)
    tg = np.linspace(0., 1., n).astype(np.float64)
    g = np.linspace(0., 1., initial_size).astype(np.float64)
    temp_domain_gamma = tg[parametrization_array[1]]

    tp_temp = tp[parametrization_array[1]]
    py_temp = py[parametrization_array[1]]
    tq_temp = tq[parametrization_array[1]]
    qy_temp = qy[parametrization_array[1]]

    # Find coarsest solution
    current_iteration = 0
    gamma_domain, gamma_range, energy = shapedist.elastic_n_2.find_gamma(
        tp[parametrization_array[0]], py[parametrization_array[0]],
        qy[parametrization_array[0]], tg[parametrization_array[0]],
        g, n_2_precision, n_2_precision, energy_dot)

    count = 0
    i = 0
    cont = True
    previous_path = np.zeros(parametrization_size, dtype=np.float64)
    while i < parametrization_size:
        previous_path[i] = interp(temp_domain_gamma[i], gamma_domain, gamma_range, 0, previous_n)
        i = i + 1

    neighborhood_array = np.zeros((parametrization_size, 2), dtype=np.int16)
    neighborhood_array = neighborhood_array + 8
    gamma_interval = 1 /(parametrization_size-1)
    strip_height = 16
    upper_bound, lower_bound = calculate_search_area(temp_domain_gamma,
                                                     temp_domain_gamma, previous_path, strip_height, parametrization_size)

    # Iteratively find the medium coarse solution
    while cont:

        path, shape_energy = iteration(tp_temp, tq_temp, py_temp, qy_temp, temp_domain_gamma, upper_bound, lower_bound,
              neighborhood_array, parametrization_size, energy_dot)

        outside_boundary = (path[:-1] > upper_bound[:-1]).any() or (path[1:] < lower_bound[1:]).any()
        pushing_boundary = (np.abs((upper_bound[:-1] - path[:-1])) < gamma_interval).any() or \
                                 (np.abs((lower_bound[1:] - path[1:])) < gamma_interval).any()
        # print(outside_boundary, pushing_boundary)
        if count >= 4:
            raise RuntimeWarning("Solution could not converge after 4 iterations.")
        cont = pushing_boundary or outside_boundary
        # print(pushing_boundary, outside_boundary)
        # plt.plot(temp_domain_gamma, path, ".-r")
        # plt.plot(temp_domain_gamma, upper_bound, "-y")
        # plt.plot(temp_domain_gamma, lower_bound, "-y")
        # plt.show()
        count = count + 1

        # print(count)
        if pushing_boundary:
            upper_bound, lower_bound = calculate_search_area(temp_domain_gamma,
                                                             temp_domain_gamma, path, strip_height,
                                                             parametrization_size)
            strip_height = strip_height + 4

    # Calculate super fine solution, if needed
    previous_n = parametrization_size
    if parametrization_array.shape[0] == 2:
        current_iteration = 1
    elif parametrization_array.shape[0] == 3:

        strip_height = 8
        current_iteration = 2
        domain_gamma = tg[parametrization_array[2]]

        tp_temp = tp[parametrization_array[2]]
        py_temp = py[parametrization_array[2]]
        tq_temp = tq[parametrization_array[2]]
        qy_temp = qy[parametrization_array[2]]
        parametrization_size = tp_temp.size
        neighborhood_array_final = np.zeros((parametrization_size, 2), dtype=np.int64)
        neighborhood_array_final = neighborhood_array_final + 8
        previous_path = np.zeros(parametrization_size, dtype=np.float64)
        i = 0

        while i < parametrization_size:
            previous_path[i] = interp(domain_gamma[i], temp_domain_gamma, path, 0, previous_n)
            i = i + 1

        upper_bound, lower_bound = calculate_search_area(domain_gamma, domain_gamma,
                                                         previous_path, strip_height, parametrization_size)

        path, shape_energy = iteration(tp_temp, tq_temp, py_temp, qy_temp, domain_gamma, upper_bound, lower_bound,
                                       neighborhood_array_final, parametrization_size, energy_dot)

    return tg[parametrization_array[current_iteration]], path, shape_energy


@jit(cache=True, nopython=True, fastmath=True)
def calculate_search_area(new_domain, prev_domain, prev_path, strip_height, parametrization_size):

    upper_bound_temp = np.zeros((prev_domain.size, 2), dtype=np.float64)
    lower_bound_temp = np.zeros((prev_domain.size, 2), dtype=np.float64)
    upper_bound_temp[:, 0] = prev_domain
    upper_bound_temp[:, 1] = prev_path
    lower_bound_temp[:, 0] = prev_domain
    lower_bound_temp[:, 1] = prev_path
    theta = pi/6
    rotation_matrix = np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))
    unit_up = np.zeros(2, np.float64)
    unit_up[1] = 1

    upper_bound_temp = upper_bound_temp + (rotation_matrix @ unit_up) * strip_height / parametrization_size
    lower_bound_temp = lower_bound_temp - (rotation_matrix @ unit_up) * strip_height / parametrization_size

    upper_bound = np.zeros(parametrization_size, dtype=np.float64)
    lower_bound = np.zeros(parametrization_size, dtype=np.float64)

    for i in range(parametrization_size):
        if new_domain[i] > upper_bound_temp[-1, 0]:
            upper_bound[i] = 1.1
        else:
            upper_bound[i] = interp(new_domain[i], upper_bound_temp[:, 0], upper_bound_temp[:, 1], 0, parametrization_size)
    for i in range(parametrization_size):
        if new_domain[i] < lower_bound_temp[0, 0]:
            lower_bound[i] = -0.1
        else:
            lower_bound[i] = interp(new_domain[i], lower_bound_temp[:, 0], lower_bound_temp[:, 1], 0, parametrization_size)

    upper_bound = running_mean(upper_bound, 6)
    lower_bound = running_mean(lower_bound, 6)
    #

    # return prev_path_interp + (rotation_matrix @ unit_up) * strip_height / parametrization_size,\
    #        prev_path_interp - (rotation_matrix @ unit_up) * strip_height / parametrization_size

    return upper_bound, lower_bound


@jit(cache=True, nopython=True, fastmath=True)
def iteration(tp_temp, tq_temp, py_temp, qy_temp, temp_domain_gamma, upper_bound, lower_bound,
              neighborhood_array, parametrization_size, energy_dot):
    # Linear Iteration
    path = np.zeros(parametrization_size, dtype=np.float64)

    m = parametrization_size
    n = parametrization_size

    min_energy_values = np.zeros((n, n), dtype=np.float64)
    path_nodes = np.zeros((n, n, 2), dtype=np.int16)
    gamma_interval = 1 / (m - 1)
    min_energy_values[0][0] = integrate(tp_temp, tq_temp, py_temp, qy_temp, 0, 1, 0, 1, gamma_interval, energy_dot)

    path_nodes[1][1][0] = 0
    path_nodes[1][1][1] = 0
    i, j, k, l = 1, 1, 1, 1
    val = 0
    val2 = 0

    while i < n - 1:
        # val = interp(tp_temp[i], prev_temp_domain_gamma,
        #              gamma_range, 0, previous_n)
        j = floor(lower_bound[i] / gamma_interval)
        if j <= 0:
            j = 1
        while j < m - 1 and j * gamma_interval < upper_bound[i]:
            min_energy_values[i][j] = integrate(tp_temp, tq_temp, py_temp, qy_temp, 0, i, 0, j,
                                                gamma_interval, energy_dot)

            k = i - neighborhood_array[i][0]
            if k <= 0:
                k = 1
            minimum = min_energy_values[i][j]
            while k < i:
                # val2 = interp(tp_temp[k],
                #               prev_temp_domain_gamma,
                #               gamma_range, 0, previous_n)

                l = j - neighborhood_array[i][1]
                if l <= floor(lower_bound[i] / gamma_interval):
                    l = floor(lower_bound[i] / gamma_interval)
                if l <= 0:
                    l = 1
                while l < j and l * gamma_interval < upper_bound[k]:
                    e = min_energy_values[k, l] + integrate(tp_temp, tq_temp, py_temp, qy_temp,
                                                            k, i, l, j,
                                                            gamma_interval, energy_dot)
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
    min_energy_values[i][j] = integrate(tp_temp, tq_temp, py_temp, qy_temp, 0, i, 0, j,
                                        gamma_interval, energy_dot)

    k = i - neighborhood_array[i][0]
    if k <= 0:
        k = 1
    minimum = min_energy_values[i][j]
    while k < i:
        # val2 = interp(tp_temp[k],
        #               prev_temp_domain_gamma,
        #               gamma_range, 0, previous_n)
        l = j - neighborhood_array[i][1]
        if l <= floor(lower_bound[i] / gamma_interval):
            l = floor(lower_bound[i] / gamma_interval)
        if l <= 0:
            l = 1
        while l < j and l * gamma_interval < upper_bound[k]:
            e = min_energy_values[k, l] + integrate(tp_temp, tq_temp, py_temp, qy_temp,
                                                    k, i, l, j,
                                                    gamma_interval, energy_dot)
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
    return path, min_energy_values[n-1][m-1]
