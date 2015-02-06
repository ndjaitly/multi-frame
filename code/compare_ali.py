import sys

file1 = sys.argv[1]
file2 = sys.argv[2]

sys.stderr.write(file1)
sys.stderr.write("\n")
sys.stderr.write(file2)
sys.stderr.write("\n")

f1 = open(file1, 'r')
lines1 = f1.readlines()
f1.close()

f2 = open(file2, 'r')
lines2 = f2.readlines()
f2.close()


num_frames, num_diffs = 0, 0

for file_num, (line1, line2) in enumerate(zip(lines1, lines2)):
    line1 = line1.rstrip('\n').rstrip(' ')
    line2 = line2.rstrip('\n').rstrip(' ')

    num_diffs_cur = 0
    if line1 != line2:
        parts1 = line1.rstrip('\n').split()
        parts2 = line2.rstrip('\n').split()

        num_diffs_cur = 0 
        for i in range(max(len(parts1), len(parts2))):
            #sys.stderr.write("%d "%i)
            #if i < len(parts1):
                #sys.stderr.write(parts1[i] + "\t")
            #else:
                #sys.stderr.write(" \t")
            #if i < len(parts2):
                #sys.stderr.write(parts2[i])
            #else:
                #sys.stderr.write(" ")
            if parts1[i] != parts2[i]:
                num_diffs_cur += 1
                #sys.stderr.write(" D")
            #sys.stderr.write("\n")
    num_frames += (len(parts1)-1)
    num_diffs += num_diffs_cur
    if num_diffs_cur != 0:
        sys.stderr.write('File # %d, # of diffs = %d, # of frames = %d\n'%(\
                                   file_num, num_diffs, num_frames))

