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


# One "step" in the trapezoidal rule, p and q are arrays
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
    while counter < n - 1:
        k = 0
        while k < n:
            minimum = np.inf
            j = 0
            while j < n:
                E = f_linear(counter, counter+1, gamma[k], gamma[j], tg, tp, tq, p, q) + min_energy_values[counter][j]
                if E < minimum:
                    minimum = E
                    val = j
                j = j + 1
            min_energy_values[counter][k] = minimum
            aux_array[counter][k] = val
            k = k + 1
        counter = counter + 1
    print(min_energy_values)
    print(min(min_energy_values[len(min_energy_values) - 2]), "energy")
    return min_energy_values, aux_array
    # min(min_energy_values[len(min_energy_values)-2])


def find_gamma(tp, tq, p, q, tg, gamma):
    min_energy_values, aux_array = find_energy(tp, tq, p,q,tg,gamma)
    min_energy = min(min_energy_values[len(min_energy_values)-2])
    index_min = np.argmin(min_energy_values[len(min_energy_values)-2])
    print(index_min, "a")
    n = len(gamma)
    print(aux_array)
    path = []
    counter = n-2
    minumum = np.inf
    while counter > 0:
        minimum = np.inf
        index = index_min
        path.append(tg[int(aux_array[counter][index])])
        index = int(aux_array[counter][index])
        counter = counter - 1
    print(path)
    return path



