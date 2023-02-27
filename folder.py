import os


for i in range(2, 6):
    for j in range(1, 26):
        for k in range(1, 6):
            folder = 'Table_' + str(i) + '/' + 'Sample_' + \
                str(j) + '/' + str(k) + '_rep'
            os.mkdir(folder)
