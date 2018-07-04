def convert(filename):
    f = open(filename, "r")

    data = f.read()
    f.close()

    data = data.strip().split("\n")
    header_list = []
    data_list = []
    error_list = []
    i = 0
    while i < len(data) // 5:
        header_list.append(data[i * 5])
        data_list.append((float(data[i * 5 + 2][29:35]) + float(data[i * 5 + 3][29:35]) + float(data[i * 5 + 4][29:35]))/3)
        error_list.append(data[i * 5 + 1].split(" ")[4])
        i = i + 1

    print(data_list)
    f = open(filename+"_result.csv", "w")
    j = 0
    base = 0
    for i in header_list:
        f.write(str(i) + "," + str(data_list[base])+ "," + error_list[base] + ",\n" )
        base = base + 1
    f.close()


convert("i_result.txt")
convert("b_result.txt")
convert("s_result.txt")