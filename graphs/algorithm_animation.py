import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
from testing import examples as ex
from scipy.io import loadmat
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.patches as patches


def interp(t, x, y, lower, upper):
    i = 0
    while lower < upper:
        i = lower + (upper - lower) // 2
        val = x[i]
        if t == val:
            break
        elif t > val:
            if lower == i:
                break
            lower = i
        elif t < val:
            upper = i
    return (t - x[i]) * (y[i + 1] - y[i]) / (x[i + 1] - x[i]) + y[i]

def integrate(tp, tq, py, qy, gamma, k, i, l, j):
    e = 0
    a = k
    while a < i:
        gammak_1 = gamma[l] + (tp[a] - tp[k]) * (gamma[j] - gamma[l]) / (tp[i] - tp[k])
        gammak_2 = gamma[l] + (tp[a+1] - tp[k]) * (gamma[j] - gamma[l]) / (tp[i] - tp[k])
        e = e + (0.5 * (py[a] - interp(gammak_1, tq, qy, 0, tq.size)) ** 2
                         + 0.5 * (py[a+1] - interp(gammak_2, tq, qy, 0, tq.size)) ** 2) * (tp[a+1] - tp[a]) * 0.5
        a = a + 1
    return e

def graph(actual_gamma, new_path_seg, old_path_seg, i):
    plt.clf()
    plt.plot(t, actual_gamma, ".-", color="#FC5C5C")
    plt.plot(new_path_seg[:,0], new_path_seg[:,1], ".-", color="#FF7F00")
    plt.plot(old_path_seg[:0], old_path_seg[:1], color="#00C7A9")
    plt.gca().add_patch(
        patches.Rectangle(
            (0, 0),  # (x,y)
            new_path_seg[-1][0],  # width
            new_path_seg[-1, 1],  # height
        )
    )
    plt.xticks([])
    plt.yticks([])
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().set_xlim([0, 1])
    plt.gca().set_ylim([0, 1])
    plt.gca().spines['bottom'].set_color('#FFFFFF')
    plt.gca().spines['left'].set_color('#FFFFFF')
    plt.gca().axhline(y=0, color='#FFFFFF')
    plt.gca().axvline(x=0, color='#FFFFFF')
    plt.savefig('./animation_frames_n_4/frame' + str(i) + '.png', transparent=True,dpi =  300)
def find_path():
    path = np.zeros((n, 2), dtype=np.int16)
    path[0][0] = n-1
    path[0][1] = m-1
    i = 0
    while path[i][0] != 0 or path[i][1] != 0:
        result = path_nodes[path[i][0]][path[i][1]]
        path[i+1][0] = result[0]
        path[i+1][1] = result[1]
        i = i + 1
    gamma_range = np.zeros(n)
    i = 1
    previous = 1
    previousIndex = n-1
    j = 0
    gamma_range[path[0][0]] = gamma[path[0][1]]
    while i < path.size // 2 and previousIndex != 0:
        gamma_range[path[i][0]] = gamma[path[i][1]]
        if previousIndex - path[i][0] > 1:
            j = 0
            step_size = (previous - gamma[path[i][1]]) / (previousIndex - path[i][0])
            while j < previousIndex - path[i][0]:
                gamma_range[previousIndex - j] = previous - j * step_size
                j = j + 1
        previousIndex = path[i][0]
        previous = gamma[path[i][1]]
        i = i + 1
def find_gamma(p, q, g):
    tp, tq, py, qy, tg, gamma = p[0], q[0], p[1], q[1], g[0], g[1]
    m = gamma.size
    n = tp.size
    min_energy_values = np.zeros((n, m), dtype=np.float64)
    path_nodes = np.zeros((n, m, 2), dtype=np.int16)

    min_energy_values[0][0] = (0.5 * (py[0] - interp(gamma[0], tq, qy, 0, tq.size)) ** 2
                         + 0.5 * (py[1] - interp(gamma[1], tq, qy, 0, tq.size)) ** 2) * (tp[1] - tp[0]) * 0.5
    path_nodes[1][1][0] = 0
    path_nodes[1][1][1] = 0
    i, j, k, l = 1, 1, 1, 1

    while i < n-1:
        j = 1
        while j < m-1:
            min_energy_values[i][j] = integrate(tp, tq, py, qy, gamma, 0, i, 0, j)
            k = 1
            minimum = min_energy_values[i][j]
            while k < i:
                l = 1
                while l < j:
                    e = min_energy_values[k, l] + integrate(tp, tq, py, qy, gamma, k, i, l, j)
                    if e < minimum:
                        minimum = e
                        path_nodes[i][j][0] = k
                        path_nodes[i][j][1] = l
                    l = l + 1
                k = k + 1
            min_energy_values[i][j] = minimum
            j = j + 1
        i = i + 1

m = 256
n = 256
t = np.linspace(0.,1., m)

p = [0, 0]

q = ex.curve_example('bumps', t)[0]

x_function = InterpolatedUnivariateSpline(t, q[0])
y_function = InterpolatedUnivariateSpline(t, q[1])




test = ex.gamma_example("sine")[0](t)
test1 = ex.gamma_example("sine")[0](t)
#
# test = np.zeros(m)
#
# i = 1
# while i < m:
#     test[i] = np.random.random_sample()
#     i = i + 1
# test.sort()
# test[m-1] = 1
# test[0] = 0

p[0] = x_function(test)
p[1] = y_function(test1)

# p = np.array(np.sin(t))
# q = np.array(np.exp(t)-1)
tg = np.linspace(0.,1., 64)
gamma = np.linspace(0., 1., 64)


find_gamma(np.array([t, p[1]]), np.array([t, q[1]]), np.array(tg, gamma))
print("hi")

































