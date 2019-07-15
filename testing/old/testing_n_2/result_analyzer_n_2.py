f = open("test_results_n_2_raw.txt", "r")
text = f.read()
text = text.split("\n\n\n")
f.close()

counter = 0


f2 = open("test_cases_n_2", "r")
text2 = f2.read().split("\n")
f2.close()
f = open("test_results_n_2.txt", "w")
for i in text:
    y = i.split("\n")
    if counter % 6 == 0:
        x = text2[counter//6].split(" ")
        m = x[1]
        curve_type = x[0]
        height1 = x[2]
        height2 = x[3]
        f.write(curve_type + " " + m + " " + height1 + " " + height2 + "\n")
        f.write(y[0] + "\n")
    else:
        f.write(y[2] + "\n")
    counter = counter + 1
    print(counter)
f.close()
