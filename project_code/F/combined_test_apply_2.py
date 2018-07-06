import numpy as np
import multiprocessing as mp
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

    number_of_threads = 1

    data = LoadFromDataBatch.ImportData(-1, 1)
    print(data.shape, type(data))
    test_data = LoadFromDataBatch.ImportData(7, 'test')
        # Parameters:
        #     picType: Positive integer between [0:10]. -1 returns all pictures.
        #     batchNumber: Positive integer between [1:5]

    nTrainingData = 1000 # number of training data
    data=data[:nTrainingData,:]  # A reduction in the set size, to test less optimal ITKrM routines without waiting hours
    test_data = test_data[:10,:]

    W_data = 32    # Width in pixels
    H_data = 32    # Height in pixels
    N_subpic = 16    # Width/Height in pixels of smaller square extracted from image.

    e = 30

    smallSet = ImportImages.ExtractSmall(data.T, W_data, H_data, N_subpic)
    testSet = ImportImages.ExtractSmall(test_data.T, W_data, H_data, N_subpic)
    print("ITKrM")
    t0 = time.time()
    dictionary = p_itkrm.itkrm(smallSet,K,S,maxit, number_of_threads) # for the apply_async (nTrainingData)
    print("execution time: {}".format(time.time() - t0))
