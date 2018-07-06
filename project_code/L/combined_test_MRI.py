import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from skimage.measure import compare_ssim
import time


import ImportImages
import OMP_fast
import LoadFromDataBatchMRI
import ITKrM_parallel_map as p_itkrm

if __name__ == '__main__':
    np.random.seed(0)

    K = 200         # Number of columns in D (Atoms)
    S = 320         # Number of used vectors (Sparsity). Amount that is NOT zero.
    maxit = 20
    N = 128**2        #Length of training examples, length of Y
    e_or_S = 1      # for error = 0, for S = 1

    pic_number = 0

    [data,test_data] = LoadFromDataBatchMRI.ImportDataMRI(-1, 1)
    print(data.shape, type(data))
    print(test_data.shape, type(test_data))

    nTrainingData = 50 # number of training data
    data=data[:nTrainingData,:]  # A reduction in the set size, to test less optimal ITKrM routines without waiting hours
    test_data = test_data[:10,:]

    W_data = 128    # Width in pixels
    H_data = 128   # Height in pixels
    N_subpic = 16    # Width/Height in pixels of smaller square extracted from image.

    S = int(S/((W_data/N_subpic)**2))
    S = 40
    e = 30
    print("S: {}".format(S))



    smallSet = ImportImages.ExtractSmall(data.T, W_data, H_data, N_subpic)
    testSet = ImportImages.ExtractSmall(test_data.T, W_data, H_data, N_subpic)
    print("ITKrM")
    t0 = time.time()
    dictionary = p_itkrm.itkrm(smallSet,K,S,maxit,2) # for the apply_async (nTrainingData)
    print("execution time: {}".format(time.time() - t0))
    print("\nOMP")
    x_sparse = OMP_fast.OMP(dictionary, testSet[:,:N], e, S, e_or_S)

    beforeImage = ImportImages.MergeSmall(testSet[:,:N], W_data, H_data, N_subpic)
    afterImage = ImportImages.MergeSmall(dictionary@x_sparse, W_data, H_data, N_subpic)


    ssim = compare_ssim(beforeImage, afterImage)
    print('ssim:',ssim)


    plt.figure('Before')
    plt.imshow(beforeImage[pic_number,:,:], cmap = 'gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.figure('After_MRI_N_data_{}_S_40'.format(N_subpic))
    plt.imshow(afterImage[pic_number,:,:], cmap = 'gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()


    b = beforeImage[pic_number,:,:]
    a = afterImage[pic_number,:,:]
    pixel_error=np.linalg.norm(b-a,ord=2)/np.linalg.norm(b,ord=2)
    print("General pixel error (p-value): {}".format(pixel_error))
