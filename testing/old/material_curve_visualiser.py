import numpy as np
import matplotlib.pyplot as plt


f = open("../data/CURVES_FeGaPd.txt", "r")

data = f.read().split("\n")
f.close()

t = data[0].split("  ")[1:]

t = [float(x) for x in t]
t = np.array(t)
t = (t - t[0]) / (t[t.size-1] - t[0])
array_of_curves = []
i = 1
while i < len(data):
    temp = data[i].split("  ")[1:]
    temp = [float(x) for x in temp]
    temp = np.array(temp)
    temp = (temp - temp.min())
    temp = temp / (temp.max())
    array_of_curves.append(temp)
    i = i + 1
array_of_curves = np.array(array_of_curves)
temp = array_of_curves[0]
gradient1 = np.gradient(temp, t)
gradient1[gradient1 == 0] = 1
srvf1 = gradient1 / np.sqrt(np.abs(gradient1))
srvf1 = (srvf1 - srvf1.min()) / (srvf1.max() - srvf1.min())
fig, ax = plt.subplots()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#FFFFFF')
ax.spines['left'].set_color('#FFFFFF')
plt.plot(t, srvf1, "#FC5C5C")
ax.set_xticks([])
ax.set_yticks([])
plt.savefig("SRVF", transparent=True)


plt.show()