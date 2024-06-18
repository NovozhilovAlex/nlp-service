file = open('help.txt', 'r')
lines = file.readlines()
file.close()

print(lines)

arr = []
for line in lines:
    line_arr = line.split(',')
    line_str = '4,'
    for i in range(len(line_arr)):
        if i != len(line_arr) - 1:
            if line_arr[i] == "":
                continue
            line_str += line_arr[i] + " "
        else:
            if line_arr[i] == '\n':
                line_str = line_str[0: len(line_str) - 1]
            else:
                line_str += line_arr[i][0:len(line_arr[i]) - 1]
    # arr.append(line_str + ' <number>' + '\n')
    arr.append(line_str + '\n')

print(arr)

file1 = open('help.csv', 'w')
file1.writelines(arr)
file1.close()

