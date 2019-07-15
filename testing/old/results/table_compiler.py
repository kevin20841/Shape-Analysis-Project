f = open("results.txt", "r")

file = f.read().split("\n")
headers = []
data = []
i = 0
f = open("N_4.csv", "w")
while i < len(file):
    if file[i] == "N_2" or file[i] == "Linear":
        print(headers)
        print(data)
        for j in headers:
            f.write(j + ",")
        f.write(",\n")
        k = 0
        while k < 5:
            a = 0
            while a < len(data):
                f.write(data[a] + ",")
                a = a + 5
            f.write(",\n")
            k = k +1
        f.close()
        headers = []
        data = []
        f = open(file[i] + ".csv", "w")

    if len(file[i].strip().split(" ")) == 2:
        headers.append(file[i])
    if len(file[i].strip().split(" ")) == 6:
        data.append(file[i].strip().split(" ")[4])
    i = i +1

for j in headers:
    f.write(j + ",")
f.write(",\n")
k = 0
while k < 5:
    a = 0
    while a < len(data):
        f.write(data[a] + ",")
        a = a + 5
    f.write(",\n")
    k = k + 1
f.close()