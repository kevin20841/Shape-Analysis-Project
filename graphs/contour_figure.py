import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image
from skimage import measure
from skimage.color import rgb2gray
import imageio

def find_curvature(a):
    dx_dt = np.gradient(a[:, 0])
    dy_dt = np.gradient(a[:, 1])

    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)

    curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
    return np.abs(curvature)

def geometric_discretization_error(b):
    T = b[:][1:] - b[:][0:-1]
    element_sizes = np.sqrt(np.sum(T ** 2, 1))
    K = np.abs(find_curvature(b))
    max_K = np.maximum(K[1:], K[:1])
    e = max_K * element_sizes ** 2
    return e

def mark_points(element_errors, tol):
    N = element_errors.size + 1
    element_markers = np.zeros(N - 1, dtype = bool)
    for i in range(element_errors.size):
        if element_errors[i] <tol:
            element_markers[i] = True
    node_markers = np.zeros(N, dtype=bool)

    node_markers[1: N-2] = np.logical_and(element_markers[0: N-3], element_markers[1: N-2])

    for k in range(1, N-1):
        if node_markers[k] and node_markers[k + 1]:
            node_markers[k + 1] = True
    return element_markers


# Construct some test data
x, y = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
r = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))
img_gray = rgb2gray(imageio.imread("puppy_sitting_2.JPG"))
# Find contours at a constant value of 0.8
contours = measure.find_contours(img_gray, 0.96)


# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

a = contours[0]

tol = 0.2
maxiter = 5
print(a.size)
for iter in range(maxiter):
    element_errors = geometric_discretization_error(a)
    markers = mark_points(element_errors, tol)
    for i in range(markers.size):
        if markers[i]:
            a[i] = [0, 0]
out = []
for i in a:
    if i[0] != 0 or i[1] != 0:
        out.append(i)
out.append(out[0])
out = np.array(out)
print(out)
print(out.size)
ax.plot(out[:, 1], out[:,0], ".-", linewidth=2)


#ax.imshow(imageio.imread("puppy_sitting_2.JPG"), interpolation='nearest', cmap=plt.cm.gray)

#ax.plot(contours[0][:, 1], contours[0][:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.savefig('dog_countour', transparent=False)
plt.show()