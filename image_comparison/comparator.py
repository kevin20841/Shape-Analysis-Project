import numpy as np

f = open("../data/shape_array.csv", "r")
test = f.read()
f.close()
test = test.strip().split("\n")
for i in range(len(test)):
    test[i] = test[i].strip().split(",")[:-1]
test = np.array(test)
test = test.astype(float)
f = open("../data/shape_array_1.csv", "r")
test1 = f.read()
f.close()

test1 = test1.strip().split("\n")
for i in range(len(test1)):
    test1[i] = test1[i].strip().split(",")[:-1]
test1 = np.array(test1)
test1 = test1.astype(float)

dif = test - test1

for i in range(100):
    for j in range(100):
        if dif[i][j] >0.01:
            print(i, j, dif[i][j], test[i][j], test1[i][j])