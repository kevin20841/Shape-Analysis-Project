import numpy as np
from numba import jit, types, float64, int16, guvectorize, generated_jit

# def integrate_2D_gamma_dot(tp, tq, py, qy, k, i, l, j, gamma_interval):
#     e = 0
#     a = k
#     while a < i:
#         gammak_1 = gamma_interval * l + (tp[a] - tp[k]) * \
#                                         (gamma_interval * j - gamma_interval * l) / (tp[i] - tp[k])
#         gammak_2 = gamma_interval * l + (tp[a+1] - tp[k]) * (gamma_interval * j - gamma_interval * l) \
#                                         / (tp[i] - tp[k])
#         gammak_dot_1 = 0
#         gammak_dot_2 = 0
#         val1 = 0.5 * (py[a][0]**2 + py[a][1] ** 2
#                       - 2 * np.sqrt(gammak_dot_1) * (py[a][0] * interp(gammak_1, tq[:, 0], qy[:, 0], 0, tq.size)
#                                                      + py[a][1] * interp(gammak_1, tq[:, 1], qy[:, 1], 0, tq.size))
#                       + gammak_dot_1 * (interp(gammak_1, tq[:, 0], qy[:, 0], 0, tq.size)**2
#                                         + interp(gammak_1, tq[:, 1], qy[:, 1], 1, tq.size)**2))
#         val2 = 0.5 * (py[a][0]**2 + py[a][1] ** 2
#                       - 2 * np.sqrt(gammak_dot_2) * (py[a][0] * interp(gammak_2, tq[:, 0], qy[:, 0], 0, tq.size)
#                                                      + py[a][1] * interp(gammak_2, tq[:, 1], qy[:, 1], 0, tq.size))
#                       + gammak_dot_2 * (interp(gammak_2, tq[:, 0], qy[:, 0], 0, tq.size)**2
#                                           + interp(gammak_2, tq[:, 1], qy[:, 1], 1, tq.size)**2))
#         e = e + (val1 + val2) * (tp[a+1] - tp[a]) * 0.5
#         a = a + 1
#     return e
