import more_itertools


def getKth_combination(n, c, k):
    return more_itertools.nth_combination(range(n), c, k)


fin = open("./include/in.txt", "r")
line = fin.readline()
fin.close()

n, r, k = line.split(' ')

# Check if k combinations exit. Also check for invalid inputs...

lst = str(getKth_combination(int(n), int(r), int(k))).replace(
    ',', ' ').replace(')', ' ').replace('(', ' ').strip()

fout = open("./include/out.txt", "w")
fout.write(lst)
fout.close()
