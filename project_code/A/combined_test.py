import numpy as np
import matplotlib.pyplot as plt

from skimage.measure import compare_ssim
import ImportImages
import OMP_fast
import LoadFromDataBatch
from time import time

import ITKrM_seq_0 as itkrm

np.random.seed(0)

K = 200         # Number of columns in D (Atoms)
S = 4         # Number of used vectors (Sparsity). Amount that is NOT zero.
maxit = 20
N = 1024        #Length of training examples, length of Y
e_or_S = 1      # for error = 0, for S = 1

pic_number = 0

data = LoadFromDataBatch.ImportData(7, 1)
test_data = LoadFromDataBatch.ImportData(7, 'test')
    # Parameters:
    #     picType: Positive integer between [0:10]. -1 returns all pictures.
    #     batchNumber: Positive integer between [1:5]

nTrainingData = 10 # number of training data
data=data[:nTrainingData,:]  # A reduction in the set size, to test less optimal ITKrM routines without waiting hours
test_data = test_data[:10,:]

W_data = 32    # Width in pixels
H_data = 32    # Height in pixels
N_subpic = 16    # Width/Height in pixels of smaller square extracted from image.

e = 30


# For testing against random dictionary
D_rand = np.random.rand(N_subpic**2, K)
for k in range(K):
    D_rand[:,k] = D_rand[:,k]/np.linalg.norm(D_rand[:,k])
#---------------------------------------------

smallSet = ImportImages.ExtractSmall(data.T, W_data, H_data, N_subpic)
testSet = ImportImages.ExtractSmall(test_data.T, W_data, H_data, N_subpic)
print("ITKrM")
t0=time()
dictionary = itkrm.itkrm(smallSet,K,S,maxit) # for the apply_async (nTrainingData)
t1=time()
print("\nOMP")
x_sparse = OMP_fast.OMP(D_rand, testSet[:,:N], e, S, e_or_S)

beforeImage = ImportImages.MergeSmall(testSet[:,:N], W_data, H_data, N_subpic)
afterImage = ImportImages.MergeSmall(D_rand@x_sparse, W_data, H_data, N_subpic)

print('\nsparseness level x:')
for i in range(int((W_data*H_data)/(N_subpic*N_subpic))):
    print(i,':', len(np.where(x_sparse[:,i]!=0)[0]))
print('Average: {}'.format(len(np.where(x_sparse!=0)[0])/testSet[:,:N].shape[1]))

ssim = compare_ssim(beforeImage, afterImage)
print('ssim:',ssim)
print("ITKrM took: {} seconds".format(t1-t0))

plt.figure('Before')
plt.imshow(beforeImage[pic_number,:,:], cmap = 'gray', vmin=0, vmax=255)
plt.tight_layout()
plt.figure('After')
plt.imshow(afterImage[pic_number,:,:], cmap = 'gray', vmin=0, vmax=255)
plt.tight_layout()
plt.show()
