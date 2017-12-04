import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

f = open("../data/shape_array.csv", "r")
test = f.read()
f.close()
test = test.strip().split("\n")
for i in range(len(test)):
    test[i] = test[i].strip().split(",")[:-1]
test = np.array(test)
test = test.astype(float)
m = np.nanmax(test)
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
heatmap = ax.pcolor(test, cmap="hot")



# plt.gca().invert_yaxis()
plt.colorbar(heatmap)
plt.figure(1)


f = open("../data/shape_array_1.csv", "r")
test = f.read()
f.close()
test = test.strip().split("\n")
for i in range(len(test)):
    test[i] = test[i].strip().split(",")[:-1]
test = np.array(test)
test = test.astype(float)
m = np.nanmax(test)
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
heatmap = ax.pcolor(test, cmap="hot")


#
# plt.gca().invert_yaxis()
plt.colorbar(heatmap)



plt.show()

# import plotly
# import plotly.graph_objs as go
#
# import pandas as pd
#
# # Read data from a csv
# z_data = pd.read_csv("../data/shape_array.csv")
#
# trace = go.Heatmap(z=test)
# data=[trace]
# plotly.offline.plot(data, filename='basic-heatmap')