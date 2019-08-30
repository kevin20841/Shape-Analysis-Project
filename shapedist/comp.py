"""
Holds all of the computation machinery of the package.
"""
from numba import jit, generated_jit
from math import floor, pi, sqrt


@jit(nopython=True)
def interpu(t, x, y, lower, upper, u):
    """
    Interpolation assuming uniform domain. Has placeholder arguments to work with @generated_jit
    Parameters
    ----------
    t : float
        The value to be interpolated
    x : array of floats
        The domain of the original curve
    y : array of floats
        The range of the original curve
    lower
    upper
    u

    Returns
    -------
    val : float
        The value interpolated
    i : float
        The index of the most near smallest domain value. Usefull for optimization.

    """
    interval = x[1] - x[0]
    i = floor(t / interval)
    if i == floor(1/interval):
        return y[i], i
    else:
        return (t - x[i]) * (y[i + 1] - y[i]) / (x[i+1] - x[i]) + y[i], i

@jit(nopython=True)
def interpn(t, x, y, lower, upper, u):
    """
    Interpolation assuming nonuniform domain. Performs a binary search.
    Parameters
    ----------
    t : float
        The value to be interpolated
    x : array of floats
        The domain of the original curve
    y : array of floats
        The range of the original curve
    lower : int
        The index of the lower portion of the search window
    upper : int
        The index of the upper portion of the search window
    u

    Returns
    -------
    val : float
        The value interpolated
    i : float
        The index of the most near smallest domain value. Usefull for optimization.

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

    if i == x.size - 1:
        val = y[i]
    else:
        val = (t - x[i]) * (y[i + 1] - y[i]) / (x[i + 1] - x[i]) + y[i]
    return val, i


@jit(nopython=True, cache=False)
def interp(t, x, y, lower, upper, u):
    """
    Wrapper function that selects which interpolation funciton to use based on if it is uniform or not.
    """
    if u:
        return interpu(t, x, y, lower, upper, u)
    else:
        return interpn(t, x, y, lower, upper, u)


@jit(nopython=True)
def norm(x, y):
    """
    Caluclates a norm between two curves

    Parameters
    ----------
    x
        The first curve
    y
        The second curve

    Returns
    -------
    s
        The value

    """
    s = 0
    for i in range(x.shape[0]):
        s = s + (x[i] - y[i])**2
    return s


@jit(nopython=True)
def srvfnorm(x, y, gammad):
    """
    Caluclates a norm between two curves, but multiplies it by a factor (for the SRVF case).

    Parameters
    ----------
    x
        The first curve
    y
        The second curve

    Returns
    -------
    s
        The value

    """
    s = 0
    for i in range(x.shape[0]):
        s = s + (x[i] - sqrt(gammad) * y[i]) ** 2
    return s


@jit(nopython=True, cache=False)
def integrate_nd(tp, tq, py, qy, k, i, l, j, gamma, energy_dot, dim, start, end, val1, val2, u):
    """
    Calculates E_i mentioned in the paper for n dimensions.
    Parameters
    ----------
    tp
    tq
    py
    qy
    k
    i
    l
    j
    gamma
    energy_dot
    dim
    start
    end
    val1
    val2
    u

    Returns
    -------

    """
    e = 0
    a = k
    gammad = (gamma[j] - gamma[l]) / (tp[i] - tp[k])
    gammak_1 = gamma[l] + (tp[a] - tp[k]) * gammad
    gammak_end = gamma[l]+ (tp[i] - tp[k]) * gammad

    for d in range(dim):
        val1[d], start[d] = interp(gammak_1, tq, qy[:, d], 0, tq.size, u)
        temp, end[d] = interp(gammak_end, tq, qy[:, d], start[d], tq.size, u)

    while a < i:
        gammak_2 = gamma[l] + (tp[a + 1] - tp[k]) * gammad
        for d in range(dim):
            val2[d], start[d] = interp(gammak_2, tq, qy[:, d], start[d], end[d] + 1, u)
        if not energy_dot:
            norm1 = norm(py[a], val1)
            norm2 = norm(py[a+1], val2)
        else:
            norm1 = srvfnorm(py[a], val1, gammad)
            norm2 = srvfnorm(py[a+1], val2, gammad)
        e = e + (norm1 + norm2) * (tp[a + 1] - tp[a]) * 0.5
        for d in range(dim):
            val1[d] = val2[d]
        a = a + 1

    return e

@jit(nopython=True, cache=False)
def integrate_1d(tp, tq, py, qy, k, i, l, j, gamma, energy_dot, dim, start, end, val1, val2, u):
    """
    Calculates E_i mentioned in paper for 1 dimension.
    Parameters
    ----------
    tp
    tq
    py
    qy
    k
    i
    l
    j
    gamma
    energy_dot
    dim
    start
    end
    val1
    val2
    u

    Returns
    -------

    """
    e = 0
    a = k
    gammak_1 = gamma[l] + (tp[a] - tp[k]) * \
               (gamma[j]- gamma[l]) / (tp[i] - tp[k])
    gammak_end = gamma[l]  + (tp[i] - tp[k]) * \
                 (gamma[j]  - gamma[l] ) / (tp[i] - tp[k])

    val1, start = interp(gammak_1, tq, qy, 0, tq.size, u)
    val2, end = interp(gammak_end, tq, qy, start, tq.size, u)

    while a < i:
        gammak_2 = gamma[l]  + (tp[a + 1] - tp[k]) * (gamma[j]  - gamma[l] ) / (
                tp[i] - tp[k])

        val2, start = interp(gammak_2, tq, qy, start, end + 1, u)
        e = e + (0.5 * (py[a] - val1) ** 2 + 0.5 *
                 (py[a + 1] - val2) ** 2) * (tp[a + 1] - tp[a]) * 0.5
        val1 = val2
        a = a + 1
    return e


@generated_jit(nopython=True, cache=False)
def integrate(tp, tq, py, qy, k, i, l, j, gamma, energy_dot, dim, start, end, val1, val2, u):
    """
    Wrapper class for integration.
    Parameters
    ----------
    tp
    tq
    py
    qy
    k
    i
    l
    j
    gamma
    energy_dot
    dim
    start
    end
    val1
    val2
    u

    Returns
    -------

    """
    if py.ndim == 1:
        return integrate_1d
    elif py.ndim == 2:
        return integrate_nd
