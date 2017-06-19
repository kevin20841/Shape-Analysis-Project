import numpy as np
import examples as ex
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
# FOLLOW PEP-8
# Compares two shapes p and q
# Use linear interpolation numpy
# Two versions: Simple trapezoidal or more precise
# Snakeviz, Numba
n = 10
t_domain = np.linspace(0.,1.,n)

b, b_deriv = ex.curve_example('hippopede',n)
g, g_deriv = ex.gamma_example('sine',0.05)


given_t = np.linspace(0.,1.,n)
given_discrete_gamma = np.linspace(0.,1.,n)

given_p_domain = t_domain
given_q_domain = t_domain

given_p_range = np.sin(t_domain* 2 * np.pi)
#given_q_range = np.sin([2 * x for x  in t_domain])
given_q_range = 2 * t_domain ** 3

#solution = np.array()


def f(t, gamma):
    return 0.5 * (np.interp(t, given_p_domain, given_p_range) - np.interp(gamma, given_q_domain, given_q_range)) ** 2


def f_simple(i, j, gamma_i, gamma_j, domain):  # Simple integral first
    t_i = domain[i]
    t_j = domain[j]
    return (f(t_j, gamma_i + (gamma_j- gamma_i)) + f(t_i, gamma_i)) * (t_j - t_i) / 2


    # integral E = 1/2 * integral from 0 to 1 of (p(t) - q(gamma(t))) ^ 2


path1 = [0] * 128
path2 = [0] * 128
counter = 0
min_energy_values = np.zeros([n, n])  # np.array(0) * len(given_discrete_gamma)
minimum = np.inf

# Find E
while counter < n-1: #len(given_t) - 1
    k = 0
    print(counter)
    while k < n:  #len(given_discrete_gamma)

        minimum = np.inf  # huge number
        j = 0
        while j < len(given_discrete_gamma):
            E = f_simple(counter, counter + 1, given_discrete_gamma[k], given_discrete_gamma[j], given_t)
            E = E + min_energy_values[counter][j]
            if E < minimum:
                minimum = E
            j = j + 1
        min_energy_values[counter][k] = minimum
        k = k + 1
    counter = counter + 1



print(min_energy_values)
print(min(min_energy_values[len(min_energy_values)-2]))
