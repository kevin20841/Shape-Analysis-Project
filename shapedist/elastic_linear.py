import numpy as np
from numba import jit, types, float64, int16


@jit([(float64, float64[:], float64[:])], cache=True, nopython=True)
def interp_linear(t, x, y):
    i = 0
    while i < x.size:
        if x[i] > t:
            i = i - 1
            break
        i = i + 1
    return (t - x[i]) * (y[i+1] - y[i]) / (x[i+1] - x[i]) + y[i]


@jit([(float64, float64[:], float64[:])], cache=True, nopython=True)
def interp(t, x, y):
    lower = 0
    upper = x.size -1
    while lower < upper:  # use < instead of <=
        i = lower + (upper - lower) // 2
        val = x[i]
        if t == val:
            break
        elif t > val:
            if lower == i:  # this two are the actual lines
                break  # you're looking for
            lower = i
        elif t < val:
            upper = i
    return (t - x[i]) * (y[i + 1] - y[i]) / (x[i + 1] - x[i]) + y[i]


@jit([types.Tuple((float64[:], float64[:], float64))(float64[:, :], float64[:, :], int16, int16)],
     cache=True, nopython=True)
def find_gamma(p, q, height, max_iteration):
    tp = p[0]
    tq = q[0]
    py = p[1]
    qy = q[1]
    path_length = tp.size
    candidates = height
    iteration = 7
    minimum = 0

    step_size = 0
    n = 2 ** iteration
    m = 2 ** (iteration-1)
    e = 0
    path = np.zeros(path_length, dtype=np.int16)
    min_energy_values = np.zeros((path_length, 2 ** max_iteration), dtype=np.float64)
    aux_array = np.zeros((path_length, 2 ** max_iteration), dtype=np.int16)

    while iteration < max_iteration:
        n = 2 ** iteration
        m = 2 ** (iteration - 1)
        counter = 0
        step_size = path_length//m
        if step_size == 0:
            m = path_length
            step_size = 1
        val = 0

        while counter < m - 1:
            k = path[(counter+1)//2] - candidates
            if k < counter + 1:
                k = counter + 1
            if (counter+1)//2 + 1 < path.size:
                temp = (path[(counter+1)//2] + path[(counter+1)//2 + 1]) // 2
            else:
                temp = path[(counter+1)//2]
            while k < 2 * (temp + candidates + 1) and k < n:
                gamma_k = 1/(n-1) * k
                minimum = np.inf
                j = k - 1
                while j > 2 * (path[counter//2] - candidates) and j >= counter:
                    gamma_j = 1/(n-1) * j
                    t_i = tp[counter * step_size]
                    t_j = tp[(counter + 1) * step_size]
                    e = (0.5 * (py[counter * step_size] - interp(gamma_j, tq, qy)) ** 2
                         + 0.5 * (py[(counter + 1) * step_size]
                                  - interp(gamma_k, tq, qy)) ** 2) * (t_j - t_i) * 0.5 \
                        + min_energy_values[counter][j]
                    if e < minimum:
                        minimum = e
                        val = j
                    j = j - 1
                min_energy_values[counter + 1][k] = minimum
                aux_array[counter][k] = val
                k = k + 1
            counter = counter + 1

        counter = m - 2
        index = n - 2
        i = 0
        minimum = np.inf
        while i < n:
            if min_energy_values[counter][i] < minimum and min_energy_values[counter][i] != 0:
                index = i
                minimum = min_energy_values[counter][i]
            i = i + 1
        path[0] = 0
        path[m-1] = n - 1
        while counter >= 0:
            path[counter] = aux_array[counter][index]
            index = aux_array[counter][index]
            counter = counter - 1
        # candidates = candidates * 2
        iteration = iteration + 1
    i = 0
    domain = np.linspace(0, 1, n)
    gamma_y = np.zeros(path_length, dtype=np.float64)
    j = path.size
    while i < path.size:
        gamma_y[i] = domain[path[i]]
        if gamma_y[i] == 1:
            j = i
        i = i + 1
    domain_gamma = np.linspace(0, 1, j + 1)
    j = j + 1
    return domain_gamma, gamma_y[:j], minimum


@jit(float64(float64[:], float64[:], float64[:]), cache=True, nopython=True)
def find_error(tg, gammar, gammat):
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

