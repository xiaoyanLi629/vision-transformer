import scipy.io
import numpy as np

# mat = scipy.io.loadmat('label.mat')
# matrix = mat['label']
# np.save(open('label.npy', 'wb'), matrix)


# x = np.load(open('label.npy', 'rb'))
for i in range(5):
    for j  in range(25):
        for k in range(5):
            old_file = 'train_annot/' + str(i+1) + '_' + str(j+1) + '_' + str(k+1) + '_label' + '.mat'
            mat = scipy.io.loadmat(old_file)
            matrix = mat['label']
            # print(matrix.shape)
            np.save(open('train_annot/' + str(i+1) + '_' + str(j+1) + '_' + str(k+1) + '_sem.npy', 'wb'), matrix)

