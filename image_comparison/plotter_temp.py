import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import stats
import image_comparison.curve_processing
from scipy.io import loadmat
from sqlalchemy.sql.functions import concat
from shapedist import elastic_linear
import numpy as np
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
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

i = 0
fig = plt.figure(1)
n = 1.*n


for i in timings:
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(n), np.log(i))
    print(slope, i)

x, = plt.loglog(n, error[1], ".-", color = "darkgreen", alpha=0.8, label="Nonuniform - Identity")

y, = plt.loglog(n, error[2], ".-", color = "mediumseagreen", alpha=0.8, label="Nonuniform - Sine")

z, = plt.loglog(n, error[4], ".-", color = "springgreen", alpha=0.8, label="Nonuniform - Steep")

x, = plt.loglog(n, error[5], ".-", color = "darkblue", alpha=0.8, label="Random - Identity")

y, = plt.loglog(n, error[7], ".-", color = "blue", alpha=0.8, label="Random - Sine")

z, = plt.loglog(n, error[8], ".-",color = "mediumslateblue", alpha=0.8, label="Random - Steep")

plt.legend(handler_map={x: HandlerLine2D(numpoints=1)})




# all_curves = loadmat('Curve_data.mat')
# print( all_curves.keys() )
# print( all_curves['MPEG7_classes'] )
# curves_128 = all_curves['MPEG7_curves_coarsened']
# curves = curves_128[10:12]
# for c in curves:
#     domain, angle_1 = image_comparison.curve_processing.convert_to_angle_function(c[0][0], c[0][1])
#     print(angle_1.max(), angle_1.min())
#     plt.plot(domain, angle_1)
#     print(angle_1)
#     plt.plot(c[0][0], c[0][1])
# plt.plot(curves[0][0][0][0], curves[0][0][1][0], ".")
# plt.plot(curves[0][0][0][1], curves[0][0][1][1], ".")
# plt.plot(curves[1][0][0][0], curves[1][0][1][0], ".")
# plt.plot(curves[1][0][0][1], curves[1][0][1][1], ".")
# # s1, theta1 = image_comparison.curve_processing.convert_to_angle_function(curves_128[0][0][0], curves_128[0][0][1])
# # s2, theta2 = image_comparison.curve_processing.convert_to_angle_function(curves_128[0][0][0], curves_128[0][0][1])
# # t = np.array(np.union1d(s1, s2))
# # angle_1_interpolation = interp1d(s1, theta1)
# # angle_2_interpolation = interp1d(s2, theta2)
# # theta1 = np.array(angle_1_interpolation(t))
# # theta2 = np.array(angle_2_interpolation(t))
# #
# # plt.plot(t, theta1, ".-r")
# # plt.plot(t, theta2, ".-g")
plt.show()
