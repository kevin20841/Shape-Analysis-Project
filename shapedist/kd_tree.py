import numpy as np
import shapedist
from numba import njit, jitclass
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt


#@njit
def ad_hoc(ind, p, q, dist):
    # find parametrization, query p onto qtree
    max_diff = 0
    for i in range(ind.shape[0] - 1):
        if ind[i+1] < ind[i] and ind[i] - ind[i+1] > max_diff:
            max_diff = ind[i] - ind[i+1]
    # print(ind.shape[0])
    # find jumps and discontinuities

    plt.plot(np.arange(0, ind.shape[0]), ind, ".r")
    plt.figure()
    i = 0
    while i < ind.shape[0]-1:
        if np.abs(ind[i+1] - ind[i]) > 30 and ind[i] - ind[i+1] != max_diff:
            start = i + 1
            j = start
            while j < ind.shape[0]-1:
                if np.abs(ind[j+1] - ind[j]) >= 30:
                    break
                j = j + 1
            end = j
            if end == ind.shape[0]:
                end = end - 1
                break
            print(start, end, end + 1 - start)
            # find current distance
            # currd = np.sum(dist[start:end + 1])
            # currd = 0
            # for l in range(start, end + 1):
            #     currd = currd + norm(p[l], q[ind[l]])

            # find first distance
            firstd = 0
            for l in range(start, end+1):
                firstd = firstd + norm(p[l], q[ind[start]])
            # find last distance
            lastd = 0
            for l in range(start, end):
                lastd = lastd + norm(p[l], q[ind[end]])

            sm = min(firstd, lastd)
            if end == ind.shape[0] - 1:
                lastc = 0
            else:
                lastc = end + 1
            if firstd == sm:
                ind[start:end + 1] = ind[start-1]
                # print("first")
            elif lastd == sm:
                ind[start:end + 1] = ind[lastc]
                # print("last")
            i = end

        i = i + 1
    plt.plot(np.arange(0, ind.shape[0]), ind, ".r")
    plt.show()
    max_diff = 0
    for i in range(ind.shape[0] - 1):
        if ind[i+1] < ind[i] and ind[i] - ind[i+1] > max_diff:
            max_diff = ind[i] - ind[i+1]

    for i in range(ind.shape[0]-1):
        if ind[i+1] < ind[i] and ind[i] - ind[i+1] != max_diff:
            ind[i+1] = ind[i]
    #
    # plt.plot(np.arange(0, ind.shape[0]), ind, ".r")
    # plt.show()
    return ind

    # max_diff = 0
    # for i in range(ind.shape[0] - 1):
    #     if ind[i+1] < ind[i] and ind[i] - ind[i+1] > max_diff:
    #         max_diff = ind[i] - ind[i+1]
    # i = 0
    # while i < ind.shape[0]-1:
    #     if ind[i+1] < ind[i] and ind[i] - ind[i+1] != max_diff:
    #         ind[i+1] = ind[i]
    #     i = i + 1
    #
    # plt.plot(np.arange(0, ind.shape[0]), ind, ".y")
    #
    # max_diff = 0
    # for i in range(ind.shape[0] - 1):
    #     if ind[i+1] < ind[i] and ind[i] - ind[i+1] > max_diff:
    #         max_diff = ind[i] - ind[i+1]
    # i = ind.shape[0] - 1
    # while i > 0:
    #     if ind[i-1] > ind[i] and ind[i-1] - ind[i] != max_diff:
    #         ind[i-1] = ind[i]
    #     i = i - 1
    # # plt.plot(np.arange(0, ind.shape[0]), ind, ".g")
    # plt.show()
    # return ind


# def ad_hoc(ind1, ind2):
#     # ind1 is p queried on qtree. ind2 is q queried on ptree
#     d = {}
#     print(ind2)
#     # inefficient for now
#     for i in range(ind2.shape[0]):
#         if ind2[i] not in d:
#             d.update({ind2[i]:[i]})
#         else:
#             d[ind2[i]].append(i)
#     print(d)
#     for i in range(ind1.shape[0]-1):
#         if ind1[i] > ind1[i+1]:
#             if i+1 not in d:
#                 print("missing value", i+1)
#             else:
#                 print("replaced", i+1)
#                 a = d[i+1].pop(0)
#                 ind1[i+1] = a
#
#     return ind1
@njit
def compute_dist(p, q, ind, t):
    s = 0
    arc = t
    for i in range(p.shape[0]-1):
        # trapezoidal method
        s = s + 0.5 * (0.5 * norm(p[i], q[ind[i]]) + 0.5 * norm(p[i + 1], q[ind[i + 1]])) * (arc[i + 1] - arc[i])
    return s


def shape_dist(p, q):
    # normalize

    N = p.shape[0]
    arclen_1 = np.sum((p[1:N, :] - p[0:N - 1, :]) ** 2, 1) ** 0.5
    arclen_1 = np.sum(arclen_1)
    p = (p - shapedist.shape_representations.calculate_com(p)) / arclen_1

    N = q.shape[0]
    arclen_2 = np.sum(np.sum((q[1:N, :] - q[0:N - 1, :]) ** 2, 1) ** 0.5)
    arclen_2 = np.sum(arclen_2)
    q = (q - shapedist.shape_representations.calculate_com(q)) / arclen_2

    t1 = shapedist.arclen_fct_values(p)
    t2 = shapedist.arclen_fct_values(q)

    [t, p, q], mask = shapedist.build_hierarchy.hierarchical_curve_discretization(p, q,
                                                                                  t1, t2,
                                                                                  2e-4,
                                                                                  True)

    # qtree = KDTree(q, leaf_size=2)
    # dist, ind1 = qtree.query(p, k=1)
    # ptree = KDTree(p, leaf_size=2)
    # dist, ind2 = ptree.query(q, k=1)
    # ind1 = np.reshape(ind1, (-1))
    # ind2 = np.reshape(ind2, (-1))
    # plt.plot(np.arange(0, ind1.shape[0]) / (ind1.shape[0] - 1), ind1 / ind1.shape[0], ".")
    # plt.plot(np.arange(0, ind2.shape[0]) / (ind2.shape[0] - 1), ind2 / ind2.shape[0], ".")
    # plt.plot(ind2 / ind2.shape[0], np.arange(0, ind2.shape[0]) / (ind2.shape[0] - 1), ".y")
    #
    # gamma = ad_hoc(ind1, ind2)
    # return gamma

    tree = KDTree(q, leaf_size=2)
    dist, ind = tree.query(p, k=1)
    ind = np.reshape(ind, (-1))
    ind = ad_hoc(ind, p, q, dist)
    return compute_dist(p, q, ind, t), ind    # (how should we define shape dist?)


@njit
def norm(x, y):
    s = 0
    for i in range(x.shape[0]):
        s = s + (x[i] - y[i])**2
    return s


@njit
def shape_dist_naive(p, q):
    dist = np.empty(q.shape[0])
    ind = np.zeros(q.shape[0], dtype=np.int64)
    for i in range(p.shape[0]):
        m = np.inf
        for j in range(i, i + 30):
            if j >= p.shape[0]:
                break
            d = norm(p[i], q[j])
            if m > d:
                m = d
                index = j
        dist[i] = m
        ind[i] = index
    return np.sum(dist), ind


def distance_matrix(curves):
    numcurves = curves.shape[0]
    distance_matrix = np.empty((numcurves, numcurves))
    for i in range(numcurves):
        tree = KDTree(curves[i], leaf_size=2)
        for j in range(i + 1, numcurves):
            dist, ind = tree.query(curves[j], k=1)
            s = np.sum(dist)
            distance_matrix[i][j] = s
            distance_matrix[j][i] = s
    return distance_matrix