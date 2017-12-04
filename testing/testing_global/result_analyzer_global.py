f = open("test_results_linear_raw.txt", "r")
text = f.read()
text = text.split("\n\n\n")
f.close()

counter = 0


f2 = open("test_cases_linear", "r")
text2 = f2.read().split("\n")
f2.close()

f = open("test_results_linear.txt", "w")
print(text2)
for i in text:
    print(i)
    y = i.split("\n")
    if counter % 6 == 0:
        x = text2[counter//6].split(" ")
        m = x[1]
        curve_type = x[0]
        f.write(curve_type + " " + m + "\n")
        f.write(y[0] + "\n")
    else:
        f.write(y[2] + "\n")
    counter = counter + 1
f.close()
