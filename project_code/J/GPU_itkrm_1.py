import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import math

import ITKrM_seq_3 as CPU_itkrm
import LoadFromDataBatch
import ImportImages
import N_timer
import time

ITKrM = gpu_itkrm

def normalize_vec(vector):
    import numpy as np
    return vector / np.linalg.norm(vector,ord=2)

def normalize_mat_col(matrix):
    return np.array([normalize_vec(matrix[:,n]) for n in range(matrix.shape[1])]).T

# @profile
def gpu_itkrm(data, K, S, maxit):
    M, N = data.shape
    D_init = np.random.randn(M, K)
    for i in range(K):
        D_init[:,i] = D_init[:,i] / np.linalg.norm(D_init[:,i], 2)
    Y = data
    I_D = np.zeros((S, N), dtype=np.int32)
    D = D_init
    TpB = 32    # ThreadsPerBlock
    BpG = math.ceil(N/32)     # BlocksPerGrid
    for t in range(maxit):
        N_timer.Timer(t, maxit)
        for n in range(N):
            I_D[:,n] = np.argpartition(np.abs(D.T@Y[:,n]), -S)[-S:]
        D_new = np.zeros((M, K))
        DtD = D.T@D
        d_D = D.reshape(-1)*1
        d_DtD = DtD.reshape(-1)*1
        d_I_D = I_D.reshape(-1)*1
        d_DtY = np.zeros(N*S)
        d_Y = Y.reshape(-1)*1
        vecproj = np.zeros(N*M*S)
        L = np.zeros(N*S*S)
        U = np.zeros(N*S*S)
        z = np.zeros(N*S)
        x = np.zeros(N*S)
        matproj = np.zeros(M*N)
        D_kernel2[TpB, BpG](M, N, K, S, d_D, d_DtD, d_I_D, d_DtY, d_Y, vecproj, L, U, z, x, matproj)
        for n in range(N):
            DtY = D[:,I_D[:,n]].T @ Y[:,n]
            signer = np.sign( DtY )
            D_new[:,I_D[:,n]] = D_new[:,I_D[:,n]] + (np.repeat(Y[:,n,None], S, axis=1) - np.repeat(matproj.reshape(N, M)[n,:,None], S, axis=1) + vecproj.reshape(N,M,S)[n,:,:])*signer
        
        scale = np.sum(D_new*D_new, axis=0)
        iszero = np.where(scale < 0.00001)[0]
        D_new[:,iszero] = np.random.randn(M, len(iszero))

        D_new = normalize_mat_col(D_new)
        D = 1*D_new
#        N_timer.log(N_timer.cont_timer(start_time,1))
#    N_timer.log("end",open_file=-1)
    return D

@cuda.jit
def D_kernel(M, N, K, S, D, DtD, I_D, Y, vecproj):
    """
    Original shapes:
        M, K = D
        K, K = DtD
        S, N = I_D
        M, N = Y
        M, S = vecproj
        M, S = D_((I_D)_n)

        M, 1 = Y_n
        M, 1 = D_((I_D)_n).T @ Y_n
    """
    n = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    n = 0
    if (n < N):
        # Vector projection
        for m in range(M):
            # Calculate D_((I_D)_n).T @ Y_n
            tmp_sum = 0
            for i in range(M):
                tmp_sum += D[i + I_D[m + n * S] * M] * Y[i + n * M]
            # Calculate D*1/diag(DtD)*DtY
            for s in range(S):
#                vecproj[m + s * M + n * M * S] = D[s + I_D[m + n * S] * M] / DtD[I_D[m + n * S] + I_D[m + n * S] * K] * tmp_sum
                vecproj[m+s*M+n*M*S] = 1 / DtD[I_D[m + n * S] + I_D[m + n * S] * K] * tmp_sum
#                vecproj[m+s*M+n*M*S] = D[s + I_D[m + n * S] * M]
            # s*M+m+n*M*S is the s-th row and m-th column of the n-th training example
            # I_D[m+n*S] + I_D[m+n*S] * K takes the diagonal elements of DtD using the n-th index set.

@cuda.jit
def D_kernel_works(M, N, K, S, D, DtD, I_D, Y, vecproj):
    n = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if (n < N):
        ### Vector projection
        for s in range(S):
            tmp_sum = 0
            # Consider saving I_D[n+s*N] in variable and access that instead. Maybe faster?
            for i in range(M):
                tmp_sum += D[I_D[n + s * N] + i * K] * Y[n + i * N]
            # Calculate D/diag(DtD)*DtY
            for m in range(M):
                vecproj[s + m*S + n*M*S] = D[I_D[n + s*N] + m*K] * tmp_sum / DtD[ I_D[n + s*N] + I_D[n + s*N]*K ]

@cuda.jit
def D_kernel2(M, N, K, S, D, DtD, I_D, DtY, Y, vecproj, L, U, z, x, matproj):
    n = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if (n < N):
        ### Vector projection
        for s in range(0, S):
           # Calculate D^T[s,:] @ Y[:,n]
           DtY[s + n*S] = 0
           for m in range(0, M):
               DtY[s + n*S] += D[I_D[n + s*N] + m*K] * Y[n + m*N]

           # Calculate D/diag(DtD)*DtY
           for m in range(0, M):
               vecproj[s + m*S + n*M*S] = D[I_D[n + s*N] + m*K] * DtY[s + n*S] / DtD[I_D[n + s*N] + I_D[n + s*N]*K]

        ### LU decomposition
        for i in range(0, S):
            for j in range(i, S):
                L[i + j*S + n*S*S] = DtD[I_D[n + i*N] + I_D[n + j*N]*K]
                for k in range(0, i):
                    L[i + j*S + n*S*S] -= L[k + j*S + n*S*S] * U[i + k*S + n*S*S]

            U[i + i*S + n*S*S] = 1
            for j in range(i+1, S):
                U[j + i*S + n*S*S] = DtD[I_D[n + j*N] + I_D[n + i*N]*K] / L[i + i*S + n*S*S]
                for k in range(0, i):
                    U[j + i*S + n*S*S] -= (L[k + i*S + n*S*S] * U[j + k*S + n*S*S]) / L[i + i*S + n*S*S]

        ### Solve system
        for i in range(0, S):
            z[i + n*S] = DtY[i + n*S] / L[i + i*S + n*S*S]
            for j in range(0, i):
                z[i + n*S] -= (L[j + i*S + n*S*S] * z[j + n*S]) / L[i + i*S + n*S*S]
        for i in range(S-1, 0-1, -1):
            x[i + n*S] = z[i + n*S]
            for j in range(i+1, S):
                x[i + n*S] -= U[j + i*S + n*S*S] * x[j + n*S]

        ### D@x
        for m in range(0, M):
           matproj[m + n*M] = 0
           for s in range(0, S):
               matproj[m + n*M] += D[I_D[n + s*N] + m*K] * x[s + n*S]



if __name__ == '__main__':
    np.random.seed(0)
    
    K = 200         # Number of columns in D (Atoms)
    S = 40         # Number of used vectors (Sparsity). Amount that is NOT zero.
    maxit = 20
    N = 1024        #Length of training examples, length of Y
    e_or_S = 1      # for error = 0, for S = 1
    
    pic_number = 0

    data = LoadFromDataBatch.ImportData(-1, 1)

    nTrainingData = 100 # number of training data
    data=data[:nTrainingData,:]  # A reduction in the set size, to test less optimal ITKrM routines without waiting hours


    W_data = 32    # Width in pixels
    H_data = 32    # Height in pixels
    N_subpic = 16    # Width/Height in pixels of smaller square extracted from image.

    smallSet = ImportImages.ExtractSmall(data.T, W_data, H_data, N_subpic)

    
    cuda.profile_start()
    t0 = time.time()
    D = ITKrM(smallSet, K, S, maxit)
    dt = time.time() - t0
    cuda.profile_stop()
    print("\n{}".format(dt))
