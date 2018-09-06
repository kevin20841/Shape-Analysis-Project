"""
Implementation of linear algorithm to determine shape elasticity as described by Bernal et al.
"""
import numpy as np
from numba import jit, float64, int16, generated_jit
import shapedist.elastic_n_2
from math import floor, pi
import matplotlib.pyplot as plt


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


def integrate_1D(tp, tq, py, qy, k, i, l, j, gamma_interval, energy_dot):
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


@generated_jit(nopython=True)
def integrate(tp, tq, py, qy, k, i, l, j, gamma_interval, energy_dot):
    if py.ndim == 1:
        return integrate_1D
    elif py.ndim == 2:
        return integrate_2D


@jit(cache=True, nopython=True)
def find_gamma(t, p, q, parametrization_array, energy_dot, gamma_tol):
    # Initialize variables
    tp = t
    tq = t
    py = p
    qy = q
    n_2_precision = 4
    strip_height = 4
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
    # Initial N^2 path
    current_iteration = 0
    gamma_domain, gamma_range, energy = shapedist.elastic_n_2.find_gamma(
        tp[parametrization_array[0]], py[parametrization_array[0]],
        qy[parametrization_array[0]], tg[parametrization_array[0]],
        g, n_2_precision, n_2_precision)

    count = 0
    i = 0
    boundary_check = False
    previous_path = np.zeros(parametrization_size, dtype=np.float64)
    while i < parametrization_size:
        previous_path[i] = interp(temp_domain_gamma[i], gamma_domain, gamma_range, 0, previous_n)
        i = i + 1
    # Create adaptive neighborhood and strip sizes
    upper_bound_temp = np.zeros((parametrization_size, 2), dtype=np.float64)
    lower_bound_temp = np.zeros((parametrization_size, 2), dtype=np.float64)
    upper_bound = np.zeros(parametrization_size, dtype=np.float64)
    lower_bound = np.zeros(parametrization_size, dtype=np.float64)
    neighborhood_array = np.zeros(parametrization_size, dtype=np.int16)
    derivative_array = np.zeros(parametrization_size, dtype=np.float64)
    derivative_array[0] = 1
    derivative_array[1:] = (previous_path[1:] - previous_path[:-1]) / (temp_domain_gamma[1:] - temp_domain_gamma[:-1])
    i = 0
    while i < parametrization_size:
        if derivative_array[i] < 1:
            neighborhood_array[i] = floor(1/derivative_array[i])
        else:
            neighborhood_array[i] = floor(derivative_array[i])
        if neighborhood_array[i] > 4:
            neighborhood_array[i] = 4
        i = i + 1
    neighborhood_array = neighborhood_array + 4
    theta = np.pi/6
    while not boundary_check:

        rotation_matrix = np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))
        unit_up = np.zeros(2, np.float64)
        unit_up[1] = 1
        upper_bound_temp[:, 0] = temp_domain_gamma
        upper_bound_temp[:, 1] = previous_path
        lower_bound_temp[:, 0] = temp_domain_gamma
        lower_bound_temp[:, 1] = previous_path

        upper_bound_temp = upper_bound_temp + (rotation_matrix @ unit_up) * strip_height / parametrization_size
        lower_bound_temp = lower_bound_temp - (rotation_matrix @ unit_up) * strip_height / parametrization_size
        i = 0
        while i < parametrization_size:
            upper_bound[i] = interp(temp_domain_gamma[i], upper_bound_temp[:, 0], upper_bound_temp[:, 1],
                                    0, parametrization_size)
            i = i + 1

        i = 0
        while i < parametrization_size:
            lower_bound[i] = interp(temp_domain_gamma[i], lower_bound_temp[:, 0], lower_bound_temp[:, 1],
                                    0, parametrization_size)
            i = i + 1
        # Linear Iteration

        current_iteration = 1

        m = parametrization_size
        n = parametrization_size

        min_energy_values = np.zeros((n, n), dtype=np.float64)
        path_nodes = np.zeros((n, n, 2), dtype=np.int16)
        gamma_interval = 1 / (m-1)
        min_energy_values[0][0] = integrate(tp_temp, tq_temp, py_temp, qy_temp, 0, 1, 0, 1, gamma_interval, energy_dot)

        path_nodes[1][1][0] = 0
        path_nodes[1][1][1] = 0
        i, j, k, l = 1, 1, 1, 1
        val = 0
        val2 = 0

        while i < n-1:
            # val = interp(tp_temp[i], prev_temp_domain_gamma,
            #              gamma_range, 0, previous_n)
            val = previous_path[i]
            j = floor(lower_bound[i] / gamma_interval)
            if j <= 0:
                j = 1
            while j < m-1 and j * gamma_interval < upper_bound[i]:
                min_energy_values[i][j] = integrate(tp_temp, tq_temp, py_temp, qy_temp, 0, i, 0, j,
                                                    gamma_interval, energy_dot)

                k = i - neighborhood_array[i]
                if k <= 0:
                    k = 1
                minimum = min_energy_values[i][j]
                while k < i:
                    # val2 = interp(tp_temp[k],
                    #               prev_temp_domain_gamma,
                    #               gamma_range, 0, previous_n)
                    val2 = previous_path[k]
                    l = j - neighborhood_array[i]
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
        i = n-1
        j = m-1
        min_energy_values[i][j] = integrate(tp_temp, tq_temp, py_temp, qy_temp, 0, i, 0, j,
                                            gamma_interval, energy_dot)

        k = i - neighborhood_array[i]
        if k <= 0:
            k = 1
        minimum = min_energy_values[i][j]
        while k < i:
            # val2 = interp(tp_temp[k],
            #               prev_temp_domain_gamma,
            #               gamma_range, 0, previous_n)
            val2 = previous_path[k]
            l = j - neighborhood_array[i]
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

        # Compare n_2 and linear gammas
        # i = 0
        # gamma_difference = 0
        # while i < n-1:
        #     # prev_gamma_value_1 = interp(temp_domain_gamma[i], prev_temp_domain_gamma, gamma_range, 0, previous_n)
        #     # prev_gamma_value_2 = interp(temp_domain_gamma[i+1], prev_temp_domain_gamma, gamma_range, 0, previous_n)
        #     prev_gamma_value_1 = previous_path[i]
        #     prev_gamma_value_2 = previous_path[i + 1]
        #
        #     gamma_difference = gamma_difference + ((prev_gamma_value_1 - path[i]) ** 2 +
        #                                            (prev_gamma_value_2 - path[i+1]) ** 2) / 2 \
        #         * (temp_domain_gamma[i + 1] - temp_domain_gamma[i])
        #
        #     i = i + 1
        #
        boundary_check_outside = (path[:-1] < upper_bound[:-1]).all() and (path[1:] > lower_bound[1:]).all()
        boundary_check_pushing = (np.abs((upper_bound - path)) > gamma_interval).all() and \
                                 (np.abs((lower_bound - path)) > gamma_interval).all()
        if count >= 4:
            raise RuntimeWarning("Solution could not converge after 4 iterations.")
        if not boundary_check_outside:
            neighborhood_array = neighborhood_array + 4
        if not boundary_check_pushing:
            theta = theta + np.pi/6
        boundary_check = boundary_check_pushing and boundary_check_outside
        count = count + 1
        # print(boundary_check_outside, boundary_check_pushing)
        # plt.plot(temp_domain_gamma, previous_path, ".-y")
        # plt.plot(temp_domain_gamma, upper_bound, "-y")
        # plt.plot(temp_domain_gamma, lower_bound, "-y")
        # plt.plot(temp_domain_gamma, path, ".-r")
        # plt.show()
        # print(count)
    return tg[parametrization_array[current_iteration]][:n], path[:n], min_energy_values[n-1][m-1]


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
