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
# def elastic_matcher(p, q, dim, parametrization=None, curve_type="coord", gamma_tol=0.0001, adaptive=True, hierarchy_tol=1, n_levels=3, max_iter=5, hierarchy_factor=2,
#                     energy_dot=False, interpolation_method='linear'):
#     t1 = None
#     t2 = None
#     if not(parametrization is None):
#         t1 = parametrization[0]
#         t2 = parametrization[1]
#     if dim == 1:
#         if hierarchy_tol == 1:
#             hierarchy_tol = 2e-5
#         original, boolean_mask, curve_hierarchy = \
#             shapedist.build_hierarchy_1D.hierarchical_curve_discretization(np.array([p, q]), hierarchy_tol,
#                                                                            n_levels=n_levels, max_iter=max_iter,
#                                                                            adaptive = adaptive,
#                                                                            interpolation_method=interpolation_method)
#
#         t_orig = original[0]
#
#         b1_orig = original[1]
#         b2_orig = original[2]
#         #     plt.figure()
#         #     plt.plot(t_orig[i], b1_orig[i], ".-r")
#         #     plt.plot(t_orig[i], b2_orig[i], ".-b")
#         # plt.show()
#         if t_orig[boolean_mask[1]].size > 500:
#             warnings.warn("Algorithm will run slowly because curves are not coarsened enough."
#                           " A larger hierarchy tolerance is recommended.", RuntimeWarning)
#         tg, gamma, energy = shapedist.elastic_linear_hierarchy.find_gamma(t_orig, b1_orig, b2_orig, boolean_mask,
#                                                                               energy_dot, gamma_tol)
#         return tg, gamma, energy, original, boolean_mask
#
#     elif dim == 2:
#         if hierarchy_tol == 1:
#             hierarchy_tol = 2e-3
#         if curve_type == "SRVF":
#             energy_dot = True
#
#         original, boolean_mask, curve_hierarchy = \
#             shapedist.build_hierarchy_2D.hierarchical_curve_discretization(np.array([p, q]),
#                                                                            t1=t1, t2=t2,
#                                                                            init_coarsening_tol=hierarchy_tol,
#                                                                            n_levels=n_levels, max_iter=max_iter,
#                                                                            adaptive=adaptive,
#                                                                            interpolation_method=interpolation_method,
#                                                                            curve_type=curve_type)
#
#         t_orig = original[0]
#         b1_orig = original[1]
#         b2_orig = original[2]
#         if t_orig[boolean_mask[1]].size > 500:
#             warnings.warn("Algorithm will run slowly because curves are not coarsened enough."
#                           " A larger hierarchy tolerance is recommended.", RuntimeWarning)
#         tg, gammay, energy = shapedist.elastic_linear_hierarchy.find_gamma(t_orig, b1_orig, b2_orig, boolean_mask,
#                                                                                    energy_dot, gamma_tol)
#         if curve_type == "SRVF" and adaptive:
#             new_b2 = np.zeros((tg.size, 2))
#             new_b2[:, 0] = np.interp(gammay, tg, b2_orig[boolean_mask[1], 0])
#             new_b2[:, 1] = np.interp(gammay, tg, b2_orig[boolean_mask[1], 1])
#             new_b1 = b1_orig[boolean_mask[1]]
#             energy = shapedist.find_shape_distance_SRVF(tg, new_b1, new_b2)
#         elif curve_type == "SRVF" and not adaptive:
#             new_b2 = np.zeros((tg.size, 2))
#             new_b2[:, 0] = np.interp(gammay, tg, b2_orig[:, 0])
#             new_b2[:, 1] = np.interp(gammay, tg, b2_orig[:, 1])
#             new_b1 = b1_orig
#             energy = shapedist.find_shape_distance_SRVF(tg, new_b1, new_b2)
#
#         return tg, gammay, energy, original, boolean_mask