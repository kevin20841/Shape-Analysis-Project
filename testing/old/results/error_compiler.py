f = open("results.txt", "r")

file = f.read().split("\n")
headers = []
data = []
i = 0
f = open("N_4_error.csv", "w")
while i < len(file):
    if file[i] == "N_2" or file[i] == "Linear":
        print(headers)
        for j in headers:
            f.write(j + ",")
        k = 0

        f.close()
        headers = []
        f = open(file[i] + "_error.csv", "w")

    if len(file[i].strip().split(" ")) == 5 or len(file[i].strip().split(" "))== 7:
        headers.append(file[i].strip().split(" ")[4])
    i = i + 1
for j in headers:
    f.write(j + ",")
f.close()