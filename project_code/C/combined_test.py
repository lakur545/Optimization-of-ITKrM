import numpy as np
import matplotlib.pyplot as plt
import ImportImages
import LoadFromDataBatch
from time import time
#import ITKrM_seq_0 as itkrm
#import ITKrM_seq_1 as itkrm
#import ITKrM_seq_2 as itkrm
import ITKrM_seq_3 as itkrm

np.random.seed(0)

K = 200         # Number of columns in D (Atoms)
S = 40         # Number of used vectors (Sparsity). Amount that is NOT zero.
maxit = 20
N = 1024        #Length of training examples, length of Y
e_or_S = 1      # for error = 0, for S = 1

#e = 30

pic_number = 0

data = LoadFromDataBatch.ImportData(7, 1)
test_data = LoadFromDataBatch.ImportData(7, 'test')
    # Parameters:
    #     picType: Positive integer between [0:10]. -1 returns all pictures.
    #     batchNumber: Positive integer between [1:5]

nTrainingData = 100 # number of training data
data=data[:nTrainingData,:]  # A reduction in the set size, to test less optimal ITKrM routines without waiting hours
test_data = test_data[:10,:]

W_data = 32    # Width in pixels
H_data = 32    # Height in pixels
N_subpic = 16    # Width/Height in pixels of smaller square extracted from image.


smallSet = ImportImages.ExtractSmall(data.T, W_data, H_data, N_subpic)

print("ITKrM")
t0=time()
dictionary = itkrm.itkrm(smallSet,K,S,maxit) # for the apply_async (nTrainingData)
t1=time()

print("ITKrM took: {} seconds".format(t1-t0))
