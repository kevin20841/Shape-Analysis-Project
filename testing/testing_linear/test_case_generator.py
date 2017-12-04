x = ["identity"]

f = open("test_case_i", "w")


for j in x:
    i = 64
    while i <= 1024:
        f.write(str(j) + " " + str(i) + "\n")
        i = i +64

f.close()