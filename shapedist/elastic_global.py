"""
Functions for finding the globally optimum solution to the shape energy problem. Call find_gamma() in order
to produce the gamma curve, and find_energy in order to produce the minimum shape energy. Do not call
find_gamma(find_energy()).

"""
import numpy as np
from numba import jit


class SizeError(Exception):
    def __init__(self, m, n):
        self.m = str(m)
        self.n = str(n)

    def __str__(self):
        return "The size (" + self.m + ") of the domain for the curves p and q" + \
               " is larger than the size (" + self.n + ") of " \
               + "the domain of gamma."


@jit(cache=True)
def find_energy(p, q, g):
    """
    Finds a the shape energy E such that E = integral from 0 to 1 of 1/2 * (p(t) - q(gamma(t))^2 is minimized.
    :param p: A 2-dimensional numpy array representing a curve P(t), with p[0] as t and p[1] as P(t)
    :param q: A 2-dimensional numpy array representing a curve Q(t), with q[0] as t and q[1] as Q(t)
    :param g: A 2-dimensional numpy array representing the parameters of the output gamma, with g[0] as the domain of
    the output gamma function and g[1] as the candidate values for the gamma function
    :return: Returns the energy, the array of minimum energy values, and the position array in that order
    """
    tp, tq, py, qy, tg, gamma = p[0], q[0], p[1], q[1], g[0], g[1]
    m = tp.size
    n = gamma.size
    if m > n:
        raise SizeError(m, n)
    min_energy_values = np.zeros([m, n])
    aux_array = np.zeros([m, n])
    counter = 0
    val = 0
    e = 0
    while counter < m - 1:
        k = counter + 1
        while k < n:
            minimum = np.inf
            j = k - 1
            while j >= counter:
                t_i = tp[counter]
                t_j = tp[counter+1]
                e = (0.5 * (np.interp(t_i, tp, py) - np.interp(gamma[j], tq, qy)) ** 2
                     + 0.5 * (np.interp(t_j, tp, py) - np.interp(gamma[k], tq, qy)) ** 2) * (t_j - t_i) * 0.5\
                    + min_energy_values[counter][j]
                if e < minimum:
                    minimum = e
                    val = j
                elif e == minimum:
                    if np.abs((gamma[k] - gamma[j]) / (tg[counter + 1] - tg[counter])-1) \
                            > np.abs((gamma[k] - gamma[val]) / (tg[counter + 1] - tg[counter])-1):
                        minimum = e
                        val = j
                j = j - 1
            min_energy_values[counter+1][k] = minimum
            aux_array[counter][k] = val
            k = k + 1
        counter = counter + 1
    i = 0
    minimum = np.inf
    while i < m:
        if min_energy_values[counter][i] < minimum and min_energy_values[counter][i]!=0:
            index = i
            minimum = min_energy_values[counter][i]
        i = i + 1
    return minimum, min_energy_values, aux_array


@jit(cache=True)
def find_gamma(p, q, g):
    """
    Finds a discrete function gamma such that the E = integral from 0 to 1 of 1/2 * (p(t) - q(gamma(t))^2 is minimized.
    :param p: A 2-dimensional numpy array representing a curve P(t), with p[0] as t and p[1] as P(t).
    :param q: A 2-dimensional numpy array representing a curve Q(t), with q[0] as t and q[1] as Q(t).
    :param g: A 2-dimensional numpy array representing the parameters of the output gamma, with g[0] as the domain of
    the output gamma function and g[1] as the candidate values for the gamma function.
    :return: Returns the y - values of the gamma function corresponding to the domain and two curves implemented. Also
    returns the minimum shape energy.
    """
    tp, tq, py, qy, tg, gamma = p[0], q[0], p[1], q[1], g[0], g[1]
    min_energy, min_energy_values, aux_array = find_energy(p, q, g)
    n = tp.size
    m = gamma.size
    path = np.zeros(n)
    counter = n-2
    index = m-1
    path[0] = 0
    path[n-1] = 1
    while counter >= 0:
        path[counter] = tg[int(aux_array[counter][index])]
        index = int(aux_array[counter][index])
        counter = counter - 1
    # func = func(tp, path)
    # path = func(tg)
    return path, min_energy


@jit(cache=True)
def find_error(tg, gammar, gammat):
    """
    Function that finds the error between two gamma curves for checking.
    :param tg: The domain of the two gamma curves.
    :param gammar: The y-values of the known gamma curve.
    :param gammat: he y-values of gamma curve to be tested.
    :return: The weighted error.
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
