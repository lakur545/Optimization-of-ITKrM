import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from skimage.measure import compare_ssim
import time


import ImportImages

import LoadFromDataBatch

import ITKrM_parallel_map as p_itkrm


if __name__ == '__main__':
    np.random.seed(0)

    K = 200         # Number of columns in D (Atoms)
    S = 40         # Number of used vectors (Sparsity). Amount that is NOT zero.
    maxit = 20
    N = 1024        #Length of training examples, length of Y
    e_or_S = 1      # for error = 0, for S = 1

    pic_number = 0
    
    number_of_threads = 4

    data = LoadFromDataBatch.ImportData(-1, 1)
    print(data.shape, type(data))
    test_data = LoadFromDataBatch.ImportData(7, 'test')

    nTrainingData = 100 # number of training data
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
    t0 = time.time()
    dictionary = p_itkrm.itkrm(smallSet,K,S,maxit,number_of_threads) # for the apply_async (nTrainingData)
    executionTime = time.time() - t0
    print("\nExecution time: {}".format(executionTime))

