print("Loading.....")
import sys
import matplotlib.pyplot as plt
import scipy.interpolate
import shapedist
from testing import examples as ex
import numpy as np
import cProfile
import pstats
from io import StringIO


# One example shapdist compuation for gamma
n = 1039  # number of points in domain

t = np.linspace(0., 1., n)
q = ex.curve_example('limacon', t)[0]
p = np.zeros(q.shape)
x_function = scipy.interpolate.CubicSpline(t, q[0])
y_function = scipy.interpolate.CubicSpline(t, q[1])


test = ex.gamma_example("steep")[0]
# test1 = ex.gamma_example("sine")[0](t)

p[0] = x_function(test(t))
p[1] = y_function(test(t))

p = p.T
q = q.T
# q = ex.curve_example("circle", t)[0].T
# p = ex.curve_example("ellipse", t)[0].T

# energy, p_new, q_new, tg, gammay = shapedist.find_shapedist(p, q, 'd', t1=t, t2=t, shape_rep=shapedist.normals)
# plt.plot(tg, gammay, "-r")

energy1, p_new, q_new, tg, gammay = shapedist.find_shapedist(p, q,'ud',  t1=t, t2=t, shape_rep=shapedist.coords)
plt.plot(tg, gammay, ".b")
plt.plot(tg, test(tg), "-r")
# for i in range(q_new.shape[1]):
#     interp_func = scipy.interpolate.CubicSpline(tg, q_new[:, i])
#     q_new[:, i] = interp_func(gammay)
plt.figure()
plt.ylim(-0.2, 0.2)
plt.xlim(-0.2, 0.2)

# plt.plot([p_new[:, 0], q_new[:, 0]], [p_new[:, 1], q_new[:, 1]], "-")

plt.plot(p_new[:, 0], p_new[:, 1], ".-b")
plt.figure()
plt.ylim(-0.2, 0.2)
plt.xlim(-0.2, 0.2)
plt.plot(q_new[:, 0], q_new[:, 1], ".-r")
plt.show()

# print(energy, energy1)
# for i in range(q_new.shape[1]):
#     interp_func = scipy.interpolate.CubicSpline(tg, q_new[:, i])
#     q_new[:, i] = interp_func(gammay)
# plt.figure()
#

#
# # get a gamma_sol who's domain matches thpe returned hierarchical domain
#



# One example shapdist compuation between two different shapes
# n = 300  # number of points in domain
#
# t = np.linspace(0., 1., n)
#
# p = ex.curve_example("ellipse", t)[0].T
# q = ex.curve_example("circle", t)[0].T
#
# ps, pa = shapedist.shape_distance_types.calculate_angle_function(p[:, 0], p[:, 1])
# qs, qa = shapedist.shape_distance_types.calculate_angle_function(q[:, 0], q[:, 1])
#
# p_angle, q_angle, tg, gammay, energy= shapedist.elastic_matcher(pa, qa, t1=ps, t2=qs)
#
# inter_func1 = scipy.interpolate.CubicSpline(t, q[:, 0])
# inter_func2 = scipy.interpolate.CubicSpline(t, q[:, 1])
# q_reparam = np.array([inter_func1(gammay[:, 0]), inter_func2(gammay[:, 0])])
# q_reparam = q_reparam.T
#
# inter_func3 = scipy.interpolate.CubicSpline(t, p[:, 0])
# inter_func4 = scipy.interpolate.CubicSpline(t, p[:, 1])
# p_reparam = np.array([inter_func3(tg), inter_func4(tg)])
# p_reparam = p_reparam.T
#
# plt.figure()
# plt.plot(tg, gammay[:, 0], ".-r")
# plt.figure()
# plt.ylim(-0.2, 0.2)
# plt.xlim(-0.2, 0.2)
# plt.plot([p_reparam[:, 0], q_reparam[:, 0]], [p_reparam[:, 1], q_reparam[:, 1]], "-")
# plt.plot(p_reparam[:, 0], p_reparam[:, 1], ".r")
# plt.plot(q_reparam[:, 0], q_reparam[:, 1], ".y")
# plt.show()

#
# print("Calculating......")
# m = 1600
# n = m
# t = np.linspace(0.,1., m)
# # t = np.random.uniform(0, 1, m-2)
# # t = np.append(t, 1)
# # t = np.insert(t, 0, 0)
# # t.sort()
#
# t2 = np.linspace(0., 1., n)
# # t_random =np.random.rand(m)
# # t_random.sort()
# # t_random[0] = 0
# # t_random[-1] = 1
# p = [0, 0]
#
# q = ex.curve_example('bumps', t)[0]
#
# x_function = scipy.interpolate.CubicSpline(t, q[0])
# y_function = scipy.interpolate.CubicSpline(t, q[1])
#
#
# test = ex.gamma_example("bumpy")[0](t)
# # test1 = ex.gamma_example("sine")[0](t)
#
# print(np.gradient(test, t).min())
# p[0] = x_function(test)
# p[1] = y_function(test)
#
# p = np.array(p)
# q = np.array(q)
# # p = ex.curve_example('hippopede', t2)[0]
# x = np.zeros([n, 2])
# y = np.zeros([m, 2])
# R_theta = np.array([[0, 1], [-1, 0]])
# print(R_theta)
# # p = np.matmul(R_theta, p)
# x[:, 0] = p[0]
# x[:, 1] = p[1]
# y[:, 0] = q[0]
# y[:, 1] = q[1]
# # print(x.shape, y.shape, t.shape, t2.shape)
# # plt.plot(t, test, ".-b")
# # plt.show()
#
#
# # tg, gammay, energy = shapedist.elastic_matcher(np.array([t, p[1]]), np.array([t, q[1]]), 1, adaptive=False)
# # tg, gammay, energy = shapedist.elastic_n_2.find_gamma(t, p[0], q[0], t, t, 4, 4, False)
# # print(shapedist.find_error(tg, ex.gamma_example("bumpy")[0](tg), gammay))
# p, q, tg, gammay, energy = shapedist.elastic_matcher(p[0], q[0], t1=t, t2=t)
#
# plt.plot(tg, gammay[:, 0], ".-r")
# plt.plot(t, test, ".-b")
# plt.show()
# #
# cProfile.runctx('tg, gammay, energy, boolean_mask = shapedist.elastic_matcher(x, y, parametrization=[t, t], dim=2, curve_type="SRVF",adaptive=False, energy_dot=True)',
#                 globals(),
#                 locals(), filename="statsfile")
# stream = StringIO()
# stats = pstats.Stats('statsfile', stream=stream).sort_stats("cumulative")
# stats.print_stats()
# print(stream.getvalue())
