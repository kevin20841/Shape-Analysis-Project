"""
Implementation of linear algorithm to determine shape elasticity as described by Bernal et al.
"""
import numpy as np
from numba import jit, types, float64, int16
np.set_printoptions(threshold=np.nan, linewidth= 400)


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
    return (t - x[i]) * (y[i + 1] - y[i]) / (x[i + 1] - x[i]) + y[i]


@jit([float64(float64[:], float64[:], float64[:],float64[:], float64[:],
              int16, int16, int16, int16)], cache=True, nopython=True)
def integrate(tp, tq, py, qy, gamma, k, i, l, j):
    e = 0
    a = k
    while a < i:
        gammak_1 = gamma[l] + (tp[a] - tp[k]) * (gamma[j] - gamma[l]) / (tp[i] - tp[k])
        gammak_2 = gamma[l] + (tp[a+1] - tp[k]) * (gamma[j] - gamma[l]) / (tp[i] - tp[k])
        e = e + (0.5 * (py[a] - interp(gammak_1, tq, qy, 0, tq.size)) ** 2
                         + 0.5 * (py[a+1] - interp(gammak_2, tq, qy, 0, tq.size)) ** 2) * (tp[a+1] - tp[a]) * 0.5
        a = a + 1
    return e


@jit([types.Tuple((float64[:], float64[:], float64))(float64[:, :], float64[:, :], float64[:, :])],
     cache=True, nopython=True)
def find_gamma(p, q, g):
    tp, tq, py, qy, tg, gamma = p[0], q[0], p[1], q[1], g[0], g[1]
    m = gamma.size
    n = tp.size
    min_energy_values = np.zeros((n, m), dtype=np.float64)
    path_nodes = np.zeros((n, m, 2), dtype=np.int16)

    min_energy_values[0][0] = (0.5 * (py[0] - interp(gamma[0], tq, qy, 0, tq.size)) ** 2
                         + 0.5 * (py[1] - interp(gamma[1], tq, qy, 0, tq.size)) ** 2) * (tp[1] - tp[0]) * 0.5
    path_nodes[1][1][0] = 0
    path_nodes[1][1][1] = 0
    i, j, k, l = 1, 1, 1, 1

    while i < n-1:
        j = 1
        while j < m-1:
            min_energy_values[i][j] = integrate(tp, tq, py, qy, gamma, 0, i, 0, j)
            k = 1
            minimum = min_energy_values[i][j]
            while k < i:
                l = 1
                while l < j:
                    e = min_energy_values[k, l] + integrate(tp, tq, py, qy, gamma, k, i, l, j)
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
    min_energy_values[i][j] = integrate(tp, tq, py, qy, gamma, 0, i, 0, j)
    k = 1
    minimum = min_energy_values[i][j]
    while k < i:
        l = 1
        while l < j:
            e = min_energy_values[k, l] + integrate(tp, tq, py, qy, gamma, k, i, l, j)
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
    while path[i][0] != 0 or path[i][1] != 0:
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