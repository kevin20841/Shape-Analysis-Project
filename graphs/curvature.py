import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image
from skimage import measure
from skimage.color import rgb2gray
import imageio
from testing import examples as ex

def find_curvature(a):
    dx_dt = np.gradient(a[0])
    dy_dt = np.gradient(a[1])

    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)

    curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
    return curvature

t = np.linspace(0., 1., 256)
q = ex.curve_example('bumps', t)[0]

curvature = find_curvature(q)
print(t, q)
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


plt.plot(t, curvature, color = "#FC5C5C")
plt.savefig('curvature.png', transparent=True,dpi =  300)
plt.show()