import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim

import KarinSchnassCopy as KSC
import ImportImages
import OMP_fast
import LoadFromDataBatch
from sequential_optimization.Iteration1 import ITKrM_optimization1 as seq_opt_1
from sequential_optimization.Iteration2 import ITKrM_optimization2 as seq_opt_2
from sequential_optimization.Iteration3 import ITKrM_optimization3 as seq_opt_3
from time import time

## Set the number of threads for numpy for true sequential performance

#import mkl
#mkl.set_num_threads(4)

##

## ITKrM settins
np.random.seed(0)

K = 200         # Number of columns in D (Atoms)
S = 4           # Number of used vectors (Sparsity). Amount that is NOT zero.
e = 30          # Error bound, how much is it allowed to be "off".
                # Read into the OMP algortihm for better understanding.
maxit = 20      # How many iterations is the dictionary updated
N = 1024        # Length of training examples, length of Y
e_or_S = 1      # for error bound = 0, for Sparsness = 1

##

## settings for using the data from CIFAR-10 https://www.cs.toronto.edu/~kriz/cifar.html
data = LoadFromDataBatch.ImportData(7, 1)
test_data = LoadFromDataBatch.ImportData(7, 'test')
    # Parameters:
    #     picType: Positive integer between [0:10]. -1 returns all pictures.
    #     batchNumber: Positive integer between [1:5]

nTrainingData = 100 # number of training data
data=data[:nTrainingData,:]  # A reduction in the set size, to test less optimal ITKrM routines without waiting hours
test_data = test_data[:10,:]

# the size of the  original pictures
W_data = 32    # Width in pixels
H_data = 32    # Height in pixels

N_subpic = 16    # Width/Height in pixels of smaller square extracted from image.

smallSet = ImportImages.ExtractSmall(data.T, W_data, H_data, N_subpic)
testSet = ImportImages.ExtractSmall(test_data.T, W_data, H_data, N_subpic)


pic_number = 0 # What picture to show from the test batch
##

## The main part

# training dictionary
print("ITKrM")
t0=time()
dictionary = KSC.itkrm(smallSet,K,S,maxit)
t1=time()
#

# Using OMP to find which set of atoms to best recreated the test batch
print("\nOMP")
x_sparse = OMP_fast.OMP(dictionary, testSet[:,:N], e, S, e_or_S)
#

# Merging subpictures of orignal testbatch and the test batch made using the found sparse solutions
beforeImage = ImportImages.MergeSmall(testSet[:,:N], W_data, H_data, N_subpic)
afterImage = ImportImages.MergeSmall(dictionary@x_sparse, W_data, H_data, N_subpic)
#


# Information and results of the process
print('\nsparseness level x:')
for i in range(int((W_data*H_data)/(N_subpic*N_subpic))):
    print(i,':', len(np.where(x_sparse[:,i]!=0)[0]))
print('Average: {}'.format(len(np.where(x_sparse!=0)[0])/testSet[:,:N].shape[1]))

ssim = compare_ssim(beforeImage, afterImage)
print('ssim:',ssim)


plt.figure('Before')
plt.imshow(beforeImage[pic_number,:,:], cmap = 'gray', vmin=0, vmax=255)
plt.tight_layout()
plt.figure('After')
plt.imshow(afterImage[pic_number,:,:], cmap = 'gray', vmin=0, vmax=255)
plt.tight_layout()


b = beforeImage[pic_number,:,:]
a = afterImage[pic_number,:,:]
pixel_error=np.linalg.norm(b-a,ord=2)/np.linalg.norm(b,ord=2)
print("General pixel error (p-value): {}".format(pixel_error))
print("ITKrM took: {} seconds".format(t1-t0))
#
