import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim

import KarinSchnassCopy as KSC
import ImportImages
import OMP_fun_1st
import OMP_stable
import OMP_amalie
import LoadFromDataBatch
import Niels_ITKrM
import Niels_ITKrM_optimized
#import Niels_ITKsM

np.random.seed(0)

K = 256         # Number of columns in D (Atoms)
S = 10           # Number of used vectors (Sparsity). Amount that is NOT zero.
e = 30
e_or_S = 1      # for error = 0, for S = 1
maxit = 20
N = 1024        #Length of training examples, length of Y

pic_number = 0

print('S:', S)
print('e:', e, '\n')

#data = np.load('grayScale32x32cars.npy')

data = LoadFromDataBatch.ImportData(7, 1)
    # Parameters:
    #     picType: Positive integer between [0:10]. -1 returns all pictures.
    #     batchNumber: Positive integer between [1:5]

data=data[:10,:]  # A reduction in the set size, to test less optimal ITKrM routines without waiting hours

W_data = 32    # Width in pixels
H_data = 32    # Height in pixels
N_subpic = 16     # Width/Height in pixels of smaller square extracted from image.


smallSet = ImportImages.ExtractSmall(data.T, W_data, H_data, N_subpic)
print("ITKrM")
dictionary = KSC.itkrm(smallSet,K,S,maxit)
print("\nOMP")
x_sparse = OMP_stable.OMP(dictionary, smallSet[:,:N], e, S, e_or_S)

beforeImage = ImportImages.MergeSmall(smallSet[:,:N], W_data, H_data, N_subpic)
afterImage = ImportImages.MergeSmall(dictionary@x_sparse, W_data, H_data, N_subpic)

print('\nsparseness level x: \n')
for i in range(int((W_data*H_data)/(N_subpic*N_subpic))):
    print(i,':', len(np.where(x_sparse[:,i]!=0)[0]))

#print('After_val', afterImage)

ssim = compare_ssim(beforeImage, afterImage)
print('ssim:',ssim)


plt.figure('Before')
plt.imshow(beforeImage[pic_number,:,:], cmap = 'gray', vmin=0, vmax=255)
plt.figure('After')
plt.imshow(afterImage[pic_number,:,:], cmap = 'gray', vmin=0, vmax=255)
plt.show()
