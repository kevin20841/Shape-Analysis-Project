import numpy as np
import matplotlib.pyplot as plt
mu= np.loadtxt("matrix_elastic.out")
mh = np.loadtxt("matrix_elastic_nonuniform.out")


mu = np.reshape(mu, (100, 100))
mh = np.reshape(mh, (100, 100))


#
# strictly_sm = True
# count = 0
# s = 0
# min_bad = np.inf
# smallest_bad_kd = np.inf
# smallest_bad_e = np.inf
# badi = 0
# badj = 0
# for i in range(100):
#     for j in range(100):
#         if matrix_kd[i][j] > matrix_elastic[i][j] and np.abs(matrix_kd[i][j] - matrix_elastic[i][j]) > 5e-04:
#             s = s + matrix_kd[i][j] - matrix_elastic[i][j]
#             strictly_sm = False
#             if min_bad > matrix_kd[i][j] - matrix_elastic[i][j]:
#                 min_bad = matrix_kd[i][j] - matrix_elastic[i][j]
#             if smallest_bad_e > matrix_elastic[i][j]:
#                 smallest_bad_e = matrix_elastic[i][j]
#                 smallest_bad_kd = matrix_kd[i][j]
#                 badi = i
#                 badj = j
#             count += 1
#             # print(i, j, matrix_kd[i][j], matrix_elastic[i][j])
#
# print("Larget elastic", np.max(matrix_elastic))
# print("Larget kd", np.max(matrix_kd))
# print("Max of kd - elastic", np.max(matrix_kd - matrix_elastic))
# print("Smallest bad e and kd", smallest_bad_e, smallest_bad_kd, badi, badj)
# print("Min of kd - elastic", min_bad)
# print("Strictly Smaller:", strictly_sm)
# print("Count:", count)
#
# print("Average Difference:", np.sum(np.abs(matrix_elastic - matrix_kd)) / (100 * 100))
# print("Average Difference for larger values:", s / count)


print(np.max(mu - mh))
plt.imshow(mu, cmap='hot', interpolation='nearest')
plt.figure()
plt.imshow(mh, cmap='hot', interpolation='nearest')
plt.figure()
plt.imshow(mh - mu, cmap="hot", interpolation="nearest")
plt.show()
