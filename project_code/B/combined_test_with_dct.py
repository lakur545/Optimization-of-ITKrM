import numpy as np
import matplotlib.pyplot as plt

from skimage.measure import compare_ssim
import ImportImages
import OMP_fast
import LoadFromDataBatch
import DCT_test
from time import time

import ITKrM_seq_0 as itkrm

np.random.seed(0)

K = 200         # Number of columns in D (Atoms)
S = 25         # Number of used vectors (Sparsity). Amount that is NOT zero.
maxit = 20
N = 1024        #Length of training examples, length of Y
e_or_S = 1      # for error = 0, for S = 1

pic_number = 0

data = LoadFromDataBatch.ImportData(7, 1)
test_data = LoadFromDataBatch.ImportData(7, 'test')
    # Parameters:
    #     picType: Positive integer between [0:10]. -1 returns all pictures.
    #     batchNumber: Positive integer between [1:5]

data=data[:40,:]  # A reduction in the set size, to test less optimal ITKrM routines without waiting hours
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
#dictionary = p_itkrm.itkrm(smallSet,K,S,maxit) # for all except the apply_async
dictionary = itkrm.itkrm(smallSet,K,S,maxit)
#dictionary = GPU_ITKrM_optimization_naive.itkrm(smallSet,K,S,maxit) # for the apply_async (nTrainingData)
t1=time()
print("\nOMP")
beforeImage = ImportImages.MergeSmall(testSet[:,:N], W_data, H_data, N_subpic)
t2=time()
x_sparse = OMP_fast.OMP(dictionary, testSet[:,:N], e, S, e_or_S)
afterImage = ImportImages.MergeSmall(dictionary@x_sparse, W_data, H_data, N_subpic)
t3=time()
print('\nsparseness level x:')
for i in range(int((W_data*H_data)/(N_subpic*N_subpic))):
    print(i,':', len(np.where(x_sparse[:,i]!=0)[0]))
print('Average: {}'.format(len(np.where(x_sparse!=0)[0])/testSet[:,:N].shape[1]))

ssim = compare_ssim(beforeImage, afterImage)
#print("ssim has been kept: {}".format(np.allclose(0.9330082894553922    ,ssim)))
plt.close("all")
plt.figure('Before')
plt.imshow(beforeImage[pic_number,:,:], cmap = 'gray', vmin=0, vmax=255)
plt.tight_layout()
plt.figure('After with ITKrM')
plt.imshow(afterImage[pic_number,:,:], cmap = 'gray', vmin=0, vmax=255)
plt.tight_layout()
#plt.show()

#plt.figure('Dictionary')
#LoadFromDataBatch.PlotPics(np.abs(dictionary.T*255))
#plt.show()

b = beforeImage[pic_number,:,:]
a = afterImage[pic_number,:,:]
pixel_error=np.linalg.norm(b-a,ord=2)/np.linalg.norm(b,ord=2)
print("General pixel error (p-value): {}".format(pixel_error))
print("ITKrM took: {} seconds".format(t1-t0))
C=0
DCT_after = []
t0 = time()
for i in range(beforeImage.shape[0]):
    A, D, B, E = DCT_test.dct_picture(beforeImage[i,:,:],S,4)
    C += B
    DCT_after += [A]
t1 = time()
print("\n")
print("OMP took: {} seconds".format(t3-t2))
print('Total ITKrM ssim: {}'.format(ssim))
print("ITKrM non-zero entries used: {}".format(S*4))
print("\n")
print("DCT took: {} seconds".format(t1-t0))
print("Total DCT ssim: {}".format(C/beforeImage.shape[0]))
print("DCT non-zero entries used: {}".format(E**2))

plt.figure('After with DCT')
plt.imshow(DCT_after[pic_number], cmap = 'gray', vmin=0, vmax=255)
plt.tight_layout()
plt.show()
