"""
Implementation of linear algorithm to determine shape elasticity as described by Bernal et al.
"""
import numpy as np
from numba import jit, types, float64, int16, generated_jit


@jit([float64(float64, float64[:], float64[:], int16, int16)], cache=False, nopython=True)
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


def integrate_1D(tp, tq, py, qy, gamma, k, i, l, j, energy_dot):
    e = 0
    a = k
    while a < i:
        gammak_1 = gamma[l] + (tp[a] - tp[k]) * (gamma[j] - gamma[l]) / (tp[i] - tp[k])
        gammak_2 = gamma[l] + (tp[a+1] - tp[k]) * (gamma[j] - gamma[l]) / (tp[i] - tp[k])
        e = e + (0.5 * (py[a] - interp(gammak_1, tq, qy, 0, tq.size)) ** 2
                         + 0.5 * (py[a+1] - interp(gammak_2, tq, qy, 0, tq.size)) ** 2) * (tp[a+1] - tp[a]) * 0.5
        a = a + 1
    return e


def integrate_2D(tp, tq, py, qy, gamma, k, i, l, j, energy_dot):
    e = 0
    a = k
    while a < i:
        gamma_derivative = (gamma[j] - gamma[l]) / (tp[i] - tp[k])
        gammak_1 = gamma[l] + (tp[a] - tp[k]) * gamma_derivative
        gammak_2 = gamma[l] + (tp[a+1] - tp[k]) * gamma_derivative

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
        e = e + (val1 + val2) * (tp[a + 1] - tp[a]) * 0.5
        a = a + 1
    return e


@generated_jit(cache=False, nopython=True)
def integrate(tp, tq, py, qy, gamma, k, i, l, j, energy_dot):
    if py.ndim == 1:
        return integrate_1D
    elif py.ndim == 2:
        return integrate_2D


@jit(cache=False, nopython=True)
def find_gamma(t, p, q, tg, gamma, width1, width2, energy_dot):
    tp = t
    tq = t
    py = p
    qy = q
    m = gamma.size
    n = tp.size
    min_energy_values = np.zeros((n, m), dtype=np.float64)
    path_nodes = np.zeros((n, m, 2), dtype=np.int16)
    min_energy_values[0][0] = integrate(tp, tq, py, qy, gamma, 0, 1, 0, 1, energy_dot)
    path_nodes[1][1][0] = 0
    path_nodes[1][1][1] = 0
    i, j, k, l = 1, 1, 1, 1

    while i < n-1:
        j = 1
        while j < m-1:
            min_energy_values[i][j] = integrate(tp, tq, py, qy, gamma, 0, i, 0, j, energy_dot)
            k = i - width1
            if k <= 0:
                k = 1
            minimum = min_energy_values[i][j]
            while k < i:
                l = j - width2
                if l <= 0:
                    l = 1
                while l < j:
                    e = min_energy_values[k, l] + integrate(tp, tq, py, qy, gamma, k, i, l, j, energy_dot)
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
    min_energy_values[i][j] = integrate(tp, tq, py, qy, gamma, 0, i, 0, j, energy_dot)
    k = i - width1
    if k <= 0:
        k = 0
    minimum = min_energy_values[i][j]
    while k < i:
        l = j - width2
        if l <= 0:
            l = 0
        while l < j:
            e = min_energy_values[k, l] + integrate(tp, tq, py, qy, gamma, k, i, l, j, energy_dot)
            if e < minimum:
                minimum = e
                path_nodes[i][j][0] = k
                path_nodes[i][j][1] = l

            l = l + 1
        k = k + 1
    min_energy_values[i][j] = minimum
    # !!
    path = np.zeros((n, 2), dtype=np.int16)
    path[0][0] = n-1
    path[0][1] = m-1
    i = 0
    while path[i][0] != 0 or path[i][1] != 0 and i + 1 < path.size:
        result = path_nodes[path[i][0]][path[i][1]]
        path[i+1][0] = result[0]
        path[i+1][1] = result[1]
        i = i + 1
    gamma_range = np.zeros(n)
    i = 1
    previous = 1
    previousIndex_domain = n-1
    previousIndex_gamma = n-1
    j = 0
    gamma_range[path[0][0]] = gamma[path[0][1]]
    while i < path.size // 2 and previousIndex_domain != 0:
        gamma_range[path[i][0]] = gamma[path[i][1]]
        if previousIndex_domain - path[i][0] > 1:
            j = 0

            while j < previousIndex_domain - path[i][0]:
                gamma_range[previousIndex_domain - j] = previous - (tg[previousIndex_domain] -
                                                                   tg[previousIndex_domain-j]) \
                                                 * (gamma[previousIndex_gamma] - gamma[path[i][1]]) / \
                                                   (tg[previousIndex_domain] - tg[path[i][0]])
                j = j + 1
        previousIndex_domain = path[i][0]
        previousIndex_gamma = path[i][1]
        previous = gamma[path[i][1]]
        i = i + 1
    return tg, gamma_range, min_energy_values[n-1][m-1]
