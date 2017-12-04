import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
from testing import examples as ex
#
# n = 128
# t = np.linspace(0.,1.,n)
#
# g = ex.gamma_example('identity')[0](t)
#
# fig, ax = plt.subplots()
# ax.plot(t, g, "y")
#
# prop = fm.FontProperties(fname='proxima-nova-regular.ttf')
# ax.set_title('"Identity" gamma curve', fontproperties=prop, size=16)
print("hi")
f = open("../data/shape_array.txt", "r")
print("hi")
x = np.loadtxt(f)
print(x[0])
temp = f.readline()
f.close()

#
# plt.show()
