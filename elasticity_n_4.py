import numpy as np
"""
Variables:
    p ---------- curve 1
    q ---------- curve 2
    tc --------- domain for both curves (if discrete must be of same length as p and q)
    tp --------- domain for p
    tq --------- domain for q
    gamma ------ range of values for possible gamma values 
    tg --------- domain for gamma (not necessarily the same as tc, (must be of same length as  gamma)
    gammad ----- one gamma function value

"""


def f_step(t, gammad, tp, tq, p, q):
    return 0.5 * (np.interp(t, tp, p) - np.interp(gammad, tq, q)) ** 2


def f_linear(i, j, gamma_i, gamma_j, domain, tp, tq, p, q):
    t_i = domain[i]
    t_j = domain[j]
    return (f_step(t_j, gamma_j, tp, tq, p, q) + f_step(t_i, gamma_i, tp, tq, p, q))\
        * (t_j - t_i) * 0.5


def find_energy(tp, tq, p, q, tg, gamma):
    n = len(gamma)
    min_energy_values = np.zeros([n, n])
    aux_array = np.zeros([n, n])
    counter = 0
    val = 0
    while counter < n - 2:
        k = 0
        while k < n:
            minimum = np.inf
            j = k
            while j >= 0:
                E = f_linear(counter, counter+1, gamma[j], gamma[k], tg, tp, tq, p, q) + min_energy_values[counter][j]
                if E < minimum:
                    minimum = E
                    val = j
                j = j - 1
            min_energy_values[counter+1][k] = minimum
            aux_array[counter+1][k] = val
            k = k + 1
        counter = counter + 1
    return min(min_energy_values[len(min_energy_values) - 2]), min_energy_values, aux_array


def find_gamma(tp, tq, p, q, tg, gamma):
    min_energy, min_energy_values, aux_array = find_energy(tp, tq, p, q, tg, gamma)
    index_min = np.argmin(min_energy_values[len(min_energy_values)-2])
    n = len(gamma)
    path = np.zeros(n)
    path[0] = 0
    path[n-1] = 1
    counter = n-2
    index = index_min
    while counter >= 1:
        path[counter] = tg[int(aux_array[counter][index])]
        index = int(aux_array[counter][index])
        counter = counter - 1
    return path

"""
gammar ----- real gamma

gammat ----- gamma to be tested
"""


def find_error(tg, gammar, gammat):
    n = len(tg)
    error = 1.0 / 2 * (tg[1] - tg[0]) * (gammar[1] - gammat[1]) ** 2 \
          + 1.0 / 2 * (tg[n-1] - tg[n-2]) * (gammar[n-1] - gammat[n-1]) ** 2
    k = 2
    while k < n-1:
        error = 1 / 2 * error + (gammar[k] - gammat[k]) ** 2 *(tg[k] - tg[k-1]) ** 2
        k = k + 1
    return error


