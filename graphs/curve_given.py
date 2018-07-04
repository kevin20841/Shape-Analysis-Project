import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
from testing import examples as ex
from scipy.io import loadmat
from scipy.interpolate import InterpolatedUnivariateSpline


f = open("../data/graphic_1.csv", "w")


all_curves = loadmat('../image_comparison/Curve_data.mat')
curves_nonuniform = all_curves['MPEG7_curves_128']
# colors =plt.rcParams['axes.color_cycle']
# print(colors)
# for i in range(10):
#     for j in range(10):
#         x = curves_nonuniform[i * 10 + j][0][0] * 1.1 + i
#         y = curves_nonuniform[i * 10 + j][0][1] * 1.1 + j
#
#         x = (x - x.min()) / (x.max() - x.min())
#         y = (y - y.min()) / (y.max() - y.min())
#         plt.fill(x + i + i * 0.1,
#                  y + j + j * 0.1, color = colors[i])
t = np.linspace(0., 1., 80)[15:-50]
q = ex.curve_example('bumps', t)[0]
print(q)

grad1 = np.gradient(q[0], t)
tangent_vector = [0, 0]
grad2 = np.gradient(q[1], t)
tangent_vector = [0, 0]
tangent_vector[0] = grad1 / np.sqrt(grad1 **2 + grad2**2)
tangent_vector[1] = grad2 / np.sqrt(grad1 **2 + grad2**2)
ax = plt.axes()
Q = plt.quiver (q[0], q[1],tangent_vector[0], tangent_vector[1], color ="#FFFFFF")

#plt.plot(t, tangent_vector[0], color = "#FC5C5C")
plt.plot(q[0][10], q[1][10], ".")
temp = tangent_vector[0] + tangent_vector[1]


i = np.argmin(temp)
print(i)
#ax.arrow(q[0][i],q[1][i], tangent_vector[0][i], tangent_vector[1][i], head_width=0.005, head_length=0.01 )
plt.plot(q[0], q[1], color = "#FC5C5C")
#plt.plot([q[0][i], q[0][i] + tangent_vector[0][i]], [q[1][i], q[1][i] + tangent_vector[1][i]])



# y = ex.gamma_example("steep")[0](x)
# curve = np.sin(x * 4)
# y2 = np.sin(y * 4)
#
# #plt.plot(x, y, color = "#FC5C5C", linewidth="2")
# plt.plot(x, curve, color = "#00C7A9", linewidth = "2")
# plt.plot(x, y2, color = "#FF7F00", linewidth = "2")
# frame = plt.gca()
plt.xticks([])
plt.yticks([])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

# plt.gca().spines['bottom'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
# plt.gca().set_xlim([0,1])
# plt.gca().set_ylim([0,1])
plt.gca().spines['bottom'].set_color('#FFFFFF')
plt.gca().spines['left'].set_color('#FFFFFF')
# plt.gca().axhline(y=0, color='#FFFFFF')
# plt.gca().axvline(x=0, color='#FFFFFF')

plt.savefig('temp.png', transparent=True,dpi =  300)


plt.show()