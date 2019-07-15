def analyze(filename):
    f = open(filename, "r")
    text = f.read().strip()
    text = text.split("\n\n\n")

    f.close()

    counter = 0

    x = filename.split("_")[0]
    f2 = open("test_case_" + x, "r")
    text2 = f2.read().strip().split("\n")
    f2.close()

    f = open(x+"_result.txt", "w")
    print(text2)
    for i in text:
        print(i)
        y = i.split("\n")
        if counter % 4 == 0:
            x = text2[counter//4].split(" ")
            m = x[1]
            curve_type = x[0]
            f.write(curve_type + " " + m + "\n")
            f.write(y[0] + "\n")
        else:
            f.write(y[2] + "\n")
        counter = counter + 1
    f.close()


analyze("sine_raw.txt")
analyze("steep_raw.txt")
analyze("bumpy_raw.txt")
analyze("flatsteep_raw.txt")