"""
Implementation of linear algorithm to determine shape elasticity as described by Bernal et al.
"""
import numpy as np
from numba import jit, types, float64, int16

import shapedist.elastic_n_2
from math import floor, pi
np.seterr(all="raise")


@jit([float64(float64, float64[:], float64[:], int16, int16)], cache=True, nopython=True)
def interp(t, x, y, lower, upper):
    """
    Linear interpolation function. Uses binary search to find which values of x to interpolate over.
    Does not work if interpolation is out of bounds

    Parameters
    ----------
    t : float
        The input of the function
    x : numpy array of floats
        The domain of the function to be interpolated
    y : numpy array of floats
        The range of the function to be interpolated

    Returns
    -------
    float
        The calculated value
    """
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



@jit([float64(float64, float64[:], float64[:], int16, int16)], cache=True, nopython=True)
def interp_uniform(t, x, y, lower, upper):
    interval = x[1] - x[0]
    i = floor(t / interval)
    return (t - x[i]) * (y[i + 1] - y[i]) / (x[i + 1] - x[i]) + y[i]



@jit([float64(float64, float64[:], float64, int16, int16)], cache=True, nopython=True)
def interp_range_only(t, y, n, lower, upper):
    """
    Linear interpolation function. Uses binary search to find which values of x to interpolate over.
    Does not work if interpolation is out of bounds

    Parameters
    ----------
    t : float
        The input of the function
    x : numpy array of floats
        The domain of the function to be interpolated
    y : numpy array of floats
        The range of the function to be interpolated

    Returns
    -------
    float
        The calculated value
    """
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
    return (t - (1/n) * i) * (y[i + 1] - y[i]) / ((1/n) * (i + 1) - (1/n) *(i)) + y[i]


@jit([float64(float64[:], float64[:], float64[:], float64[:], int16, int16, int16, int16, float64)], cache=True, nopython=True)
def integrate(tp, tq, py, qy, k, i, l, j, gamma_interval):
    e = 0
    a = k
    while a < i:
        gammak_1 = gamma_interval * l + (tp[a] - tp[k]) * \
                                        (gamma_interval * j - gamma_interval * l) / (tp[i] - tp[k])
        gammak_2 = gamma_interval * l + (tp[(a+1)] - tp[k]) * (gamma_interval * j - gamma_interval * l) \
                                        / (tp[i] - tp[k])
        e = e + (0.5 * (py[a] - interp(gammak_1, tq, qy, 0, tq.size)) ** 2
                         + 0.5 * (py[(a+1)] - interp(gammak_2, tq, qy, 0, tq.size)) ** 2) * \
                (tp[(a+1)] - tp[a]) * 0.5
        a = a + 1
    return e


@jit([types.Tuple((float64[:], float64[:], float64))(float64[:, :], float64[:, :], int16, int16, int16)], cache=True, nopython=True)
def find_gamma(p, q, neighborhood, strip_height, max_iteration):
    """
    Finds the discretized function gamma, and the minimum energy.

    Parameters
    ----------

    p : array of two arrays of floats
        The first input curve, an array with 2 elements: the domain of p as an array and the range of
        p as an array in that order. Both arrays must be the same size as q
    q : array of two arrays of floats
        The first second curve, an array with 2 elements: the domain of q as an array and the range of
        q as an array in that order. Both arrays must be the same length as p
    neighborhood : int
        The height of the adapting strip. Generally advised to be 1/3 or 1/4 of 2**max_iteration. -1 uses the value
        max_iteration * 30.
    max_iteration : int
        The resolution of the algorithm. Actual resolution is 2**max_iteration. Default value is 10. -1 uses that value

    Returns
    -------
    array of floats, array of floats, float
        The domain of gamma as an array of floats, the range of gamma as an array, of floats,
        and the minimum energy calculated as a float.

    """
    current_iteration = 4
    tp, tq, py, qy = p[0], q[0], p[1], q[1]
    n = tp.size
    path = np.zeros(n + 1, dtype=np.float64)
    i = 0
    # !!
    tg = np.linspace(0., 1., 2 ** current_iteration).astype(np.float64)
    g = np.linspace(0., 1., 2 ** current_iteration).astype(np.float64)
    domain_interval = 0
    domain_interval = tp.size // (2 ** current_iteration)
    temp1 = np.zeros((2, 2**current_iteration), dtype=np.float64)
    temp2 = np.zeros((2, 2**current_iteration), dtype=np.float64)
    temp3 = np.zeros((2, 2**current_iteration), dtype=np.float64)

    while i < 2**current_iteration:
        temp1[0][i] = tp[i * domain_interval]
        temp1[1][i] = py[i * domain_interval]
        temp2[0][i] = tq[i * domain_interval]
        temp2[1][i] = qy[i * domain_interval]
        temp3[0][i] = tg[i]
        temp3[1][i] = g[i]
        i = i + 1
    tg, gamma_range, val = shapedist.elastic_n_2.find_gamma(temp1, temp2, temp3, 4, 4)
    i = 0
    while i < gamma_range.size:
        path[i] = gamma_range[i]
        i = i + 1
    current_iteration = current_iteration + 1
    # !!
    i = 0
    min_energy_values = np.zeros((n, 2 ** max_iteration), dtype=np.float64)
    path_nodes = np.zeros((n, 2**max_iteration, 2), dtype=np.int16)
    previous_n = gamma_range.size
    rough_path = np.zeros(tp.size+1, dtype=np.int16)

    while current_iteration <= max_iteration:
        if rough_path[-2] == tp.size-1:
            break
        m = 2 ** current_iteration
        n = 2 ** current_iteration
        if n > tp.size:
            n = tp.size
        domain_interval = tp.size // n
        n = tp.size // domain_interval
        if domain_interval == 0:
            domain_interval = 1
        i = 0

        while i < n:
            rough_path[i] = domain_interval * i
            i = i + 1

        if n < rough_path.size - 1:
            rough_path[n] = tp.size-1
            n = n + 1
        gamma_interval = 1 / (m-1)
        min_energy_values[0][0] = (0.5 * (py[0] - interp(0, tq, qy, 0, tq.size)) ** 2
                                   + 0.5 * (py[1] - interp(gamma_interval, tq, qy, 0, tq.size)) ** 2) * (tp[1] - tp[0]) * 0.5
        path_nodes[1][1][0] = 0
        path_nodes[1][1][1] = 0
        i, j, k, l = 1, 1, 1, 1
        val = 0

        val2 = 0
        while i < n-1:
            val = interp_range_only(tp[rough_path[i]], path, previous_n, 0, previous_n)
            j = floor(val / gamma_interval) - strip_height
            if j <= 0:
                j = 1
            while j < m-1 and j * gamma_interval < val + strip_height * gamma_interval:
                min_energy_values[i][j] = integrate(tp, tq, py, qy, 0, rough_path[i], 0, j,gamma_interval)

                k = i - neighborhood
                if k <= 0:
                    k = 1
                minimum = min_energy_values[i][j]
                while k < i:
                    val2 = interp_range_only(tp[rough_path[k]], path, previous_n, 0,
                                             previous_n)
                    l = j - neighborhood
                    if l <= floor(val2 / gamma_interval) - strip_height:
                        l = floor(val2 / gamma_interval) - strip_height
                    if l <= 0:
                        l = 1
                    while l < j and l * gamma_interval < val2 + strip_height * gamma_interval:
                        e = min_energy_values[k, l] + integrate(tp, tq, py, qy, rough_path[k], rough_path[i], l, j,
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
        min_energy_values[i][j] = integrate(tp, tq, py, qy, 0, rough_path[i], 0, j, gamma_interval)

        k = i - neighborhood
        if k <= 0:
            k = 1
        minimum = min_energy_values[i][j]
        while k < i:
            val2 = interp_range_only(tp[rough_path[k]], path, previous_n, 0,
                                     previous_n)
            l = j - neighborhood
            if l <= floor(val2 / gamma_interval) - strip_height:
                l = floor(val2 / gamma_interval) - strip_height
            if l <= 0:
                l = 1
            while l < j and l * gamma_interval < val2 + strip_height * gamma_interval:
                e = min_energy_values[k, l] + integrate(tp, tq, py, qy, rough_path[k], rough_path[i], l, j,
                                                        gamma_interval)

                if e < minimum:
                    minimum = e
                    path_nodes[i][j][0] = k
                    path_nodes[i][j][1] = l
                l = l + 1
            k = k + 1

        min_energy_values[i][j] = minimum

        # !!

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
        previousIndex = n - 1
        j = 1
        path[path_indices[0][0]] = gamma_interval * path_indices[0][1]
        while i < path_indices.size // 2 and previousIndex != 0:
            path[path_indices[i][0]] = gamma_interval * path_indices[i][1]
            if previousIndex - path_indices[i][0] > 1:
                j = 0
                step_size = (previous - gamma_interval*path_indices[i][1]) / (previousIndex - path_indices[i][0])
                while j < previousIndex - path_indices[i][0]:
                    path[previousIndex - j] = previous - j * step_size
                    j = j + 1
            previousIndex = path_indices[i][0]
            previous = gamma_interval * path_indices[i][1]
            i = i + 1
        previous_n = n
        current_iteration = current_iteration + 1

    tg = np.linspace(0., 1., n).astype(np.float64)
    return tg, path[:n], min_energy_values[n-1][m-1]


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


@jit(float64(float64[:], float64[:], float64[:]), cache=True, nopython=True)
def find_error(tg, gammar, gammat):
    """
    Function that finds the error between two gamma curves for checking.

    Parameters
    ----------
    tg : array of floats
        The domain of the two gamma curves.
    gammar : array of floats
        The y-values of the known gamma curve.
    gammat : array of floats
        The y-values of gamma curve to be tested.

    Returns
    -------
    float
        The weighted error.
    """
    n = tg.size
    error = 1 / 2 * (tg[1] - tg[0]) * (gammar[1] - gammat[1]) ** 2 + 1 / 2 * (tg[n-1] - tg[n-2]) * (gammar[n-1] - gammat[n-1]) ** 2
    k = 2
    if n != gammar.size or n != gammat.size:
        raise IndexError
    while k < n-1:
        error = error + 1/2 * (gammar[k] - gammat[k]) ** 2 * (tg[k] - tg[k-1]) ** 2
        k = k + 1
    error = error ** (1/2)
    return error
