import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.legend_handler import HandlerLine2D
curve_type = []
timings = []
error = []
n = []
name = "uniform"
f = open("../final_data/linear_" + name + "_domain/i_result.txt_result.csv")
x = f.read().strip().split("\n")
f.close()
x = [a[:-1] for a in x]
curve_type.append(x[0].split(",")[0].split(" ")[0])
n = [a.split(",")[0].split(" ")[1] for a in x]
n = np.array(n, dtype=np.int16)
timings.append(np.array([a.split(",")[1] for a in x], dtype=np.float16))
error.append(np.array([a.split(",")[2] for a in x], dtype=np.float16))


f = open("../final_data/linear_" + name + "_domain/s_result.txt_result.csv")
x = f.read().strip().split("\n")
f.close()
x = [a[:-1] for a in x]
curve_type.append(x[0].split(",")[0].split(" ")[0])
timings.append(np.array([a.split(",")[1] for a in x], dtype=np.float16))
error.append(np.array([a.split(",")[2] for a in x], dtype=np.float16))


f = open("../final_data/linear_" + name + "_domain/b_result.txt_result.csv")
x = f.read().strip().split("\n")
f.close()
x = [a[:-1] for a in x]
curve_type.append(x[0].split(",")[0].split(" ")[0])
timings.append(np.array([a.split(",")[1] for a in x], dtype=np.float16))
error.append(np.array([a.split(",")[2] for a in x], dtype=np.float16))

name = "nonuniform"
f = open("../final_data/linear_" + name + "_domain/i_result.txt_result.csv")
x = f.read().strip().split("\n")
f.close()
x = [a[:-1] for a in x]
curve_type.append(x[0].split(",")[0].split(" ")[0])
n = [a.split(",")[0].split(" ")[1] for a in x]
n = np.array(n, dtype=np.int16)
timings.append(np.array([a.split(",")[1] for a in x], dtype=np.float16))
error.append(np.array([a.split(",")[2] for a in x], dtype=np.float16))


f = open("../final_data/linear_" + name + "_domain/s_result.txt_result.csv")
x = f.read().strip().split("\n")
f.close()
x = [a[:-1] for a in x]
curve_type.append(x[0].split(",")[0].split(" ")[0])
timings.append(np.array([a.split(",")[1] for a in x], dtype=np.float16))
error.append(np.array([a.split(",")[2] for a in x], dtype=np.float16))


f = open("../final_data/linear_" + name + "_domain/b_result.txt_result.csv")
x = f.read().strip().split("\n")
f.close()
x = [a[:-1] for a in x]
curve_type.append(x[0].split(",")[0].split(" ")[0])
timings.append(np.array([a.split(",")[1] for a in x], dtype=np.float16))
error.append(np.array([a.split(",")[2] for a in x], dtype=np.float16))

name = "random"
f = open("../final_data/linear_" + name + "_domain/i_result.txt_result.csv")
x = f.read().strip().split("\n")
f.close()
x = [a[:-1] for a in x]
curve_type.append(x[0].split(",")[0].split(" ")[0])
n = [a.split(",")[0].split(" ")[1] for a in x]
n = np.array(n, dtype=np.int16)
timings.append(np.array([a.split(",")[1] for a in x], dtype=np.float16))
error.append(np.array([a.split(",")[2] for a in x], dtype=np.float16))


f = open("../final_data/linear_" + name + "_domain/s_result.txt_result.csv")
x = f.read().strip().split("\n")
f.close()
x = [a[:-1] for a in x]
curve_type.append(x[0].split(",")[0].split(" ")[0])
timings.append(np.array([a.split(",")[1] for a in x], dtype=np.float16))
error.append(np.array([a.split(",")[2] for a in x], dtype=np.float16))


f = open("../final_data/linear_" + name + "_domain/b_result.txt_result.csv")
x = f.read().strip().split("\n")
f.close()
x = [a[:-1] for a in x]
curve_type.append(x[0].split(",")[0].split(" ")[0])
timings.append(np.array([a.split(",")[1] for a in x], dtype=np.float16))
error.append(np.array([a.split(",")[2] for a in x], dtype=np.float16))







timings = np.array(timings, dtype=np.float16)

error = np.array(error, dtype=np.float16)

fig = plt.figure(1)
area = np.pi*4
colors = ("b", "r", "g")
ax = fig.add_subplot(111)
x, = plt.plot(n, timings[0], ".-", color = "darkred", alpha=0.8, label="Uniform - Identity")

y, = plt.plot(n, timings[1], ".-", color = "red", alpha=0.8, label="Uniform - Sine")

z, = plt.plot(n, timings[2], ".-", color = "orangered", alpha=0.8, label="Uniform - Steep")

x, = plt.plot(n, timings[3], ".-", color = "darkgreen", alpha=0.8, label="Nonuniform - Identity")

y, = plt.plot(n, timings[4], ".-", color = "mediumseagreen", alpha=0.8, label="Nonuniform - Sine")

z, = plt.plot(n, timings[5], ".-", color = "springgreen", alpha=0.8, label="Nonuniform - Steep")

x, = plt.plot(n, timings[6], ".-", color = "darkblue", alpha=0.8, label="Random - Identity")

y, = plt.plot(n, timings[7], ".-", color = "blue", alpha=0.8, label="Random - Sine")

z, = plt.plot(n, timings[8], ".-",color = "mediumslateblue", alpha=0.8, label="Random - Steep")

plt.legend(handler_map={x: HandlerLine2D(numpoints=1)})
ax.set_ylabel('Time (seconds)')
ax.set_xlabel('n (number of nodes)')

fig = plt.figure(2)
area = np.pi*10
colors = ("b", "r", "g")
ax = fig.add_subplot(111)


y, = plt.plot(n, error[1], ".-", color = "red", alpha=0.8, label="Uniform - Sine")

z, = plt.plot(n, error[2], ".-", color = "orangered", alpha=0.8, label="Uniform - Steep")

y, = plt.plot(n, error[4], ".-", color = "mediumseagreen", alpha=0.8, label="Nonuniform - Sine")

z, = plt.plot(n, error[5], ".-", color = "springgreen", alpha=0.8, label="Nonuniform - Steep")

y, = plt.plot(n, error[7], ".-", color = "blue", alpha=0.8, label="Random - Sine")

z, = plt.plot(n, error[8], ".-",color = "mediumslateblue", alpha=0.8, label="Random - Steep")
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_ylabel('Error')
ax.set_xlabel('n (number of nodes)')


plt.show()

# f = open("../data/shape_array.csv", "r")
# test = f.read()
# f.close()
# test = test.strip().split("\n")
# for i in range(len(test)):
#     test[i] = test[i].strip().split(",")[:-1]
# test = np.array(test)
# test = test.astype(float)
# m = np.nanmax(test)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# fig.subplots_adjust(top=0.85)
# heatmap = ax.pcolor(test, cmap="hot")
#
# ax.set_title('Heatmap of random domain shape-distance \n values on MPEG-7 dataset')
#
#
# plt.gca().invert_yaxis()
# plt.colorbar(heatmap)
# plt.show()

# f = open("../data/shape_array_1.csv", "r")
# test1 = f.read()
# f.close()
# test1 = test1.strip().split("\n")
# for i in range(len(test1)):
#     test1[i] = test1[i].strip().split(",")[:-1]
# test1 = np.array(test1)
# test1 = test.astype(float)
# plt.imshow(test1, cmap='hot', interpolation='nearest')
#
#
# plt.show()



