import numpy as np
import matplotlib.pyplot as plt
import numba as nb
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

#@profile
def gpu_itkrm(data, K, S, maxit):
    M, N = data.shape
    D_init = np.random.randn(M, K)
    for i in range(K):
        D_init[:,i] = D_init[:,i] / np.linalg.norm(D_init[:,i], 2)
    Y = data
    I_D = np.zeros((S, N), dtype=np.int32)
    D = D_init
    TpB = 32    # ThreadsPerBlock
    BpG_N = math.ceil(N/32)     # BlocksPerGrid
    BpG_M = math.ceil(M/32)

    # Move training data to device and allocate arrays on device.
    d_Y = cuda.to_device(Y.reshape(-1)*1)
    d_vecproj = cuda.device_array(shape=(N*M*S))
    z = cuda.device_array(shape=(N*S))
    x = cuda.device_array(shape=(N*S))
    d_matproj = cuda.device_array(shape=(M*N))
    d_signer = cuda.device_array(shape=(S*N))

    for t in range(maxit):
        N_timer.Timer(t, maxit)
        for n in range(N):
            I_D[:,n] = np.argpartition(np.abs(D.T@Y[:,n]), -S)[-S:]
        D_new = np.zeros((M, K))
        DtD = D.T@D

        d_D = cuda.to_device(D.reshape(-1)*1)
        d_DtD = cuda.to_device(DtD.reshape(-1)*1)
        d_I_D = cuda.to_device(I_D.reshape(-1)*1)
        d_Dnew = cuda.to_device(D_new.reshape(-1)*1)

        k_matvecproj[TpB, BpG_N](d_D, d_DtD, d_I_D, d_Y, d_vecproj, z, x, d_matproj, d_signer)
        
        k_updateD[TpB, BpG_M](d_Dnew, d_I_D, d_Y, d_vecproj, d_matproj, d_signer)
        D_new = d_Dnew.copy_to_host()
        D_new = D_new.reshape(M, K)

        scale = np.sum(D_new*D_new, axis=0)
        iszero = np.where(scale < 0.00001)[0]
        D_new[:,iszero] = np.random.randn(M, len(iszero))

        D_new = normalize_mat_col(D_new)
        D = 1*D_new
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
def k_matvecproj(D, DtD, I_D, Y, vecproj, z, x, matproj, signer):
    n = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    # Local arrays
    # DtY = cuda.local.array(shape=S, dtype=nb.float64)
    L = cuda.local.array(shape=SS, dtype=nb.float64)
    U = cuda.local.array(shape=SS, dtype=nb.float64)
    if (n < N):
        ### Vector projection
        for i in range(0, S):
           # Calculate D^T[s,:] @ Y[:,n]
            DtY = 0
            for m in range(0, M):
                DtY += D[I_D[n + i*N] + m*K] * Y[n + m*N]

            # Calculate D/diag(DtD)*DtY
            for m in range(0, M):
                vecproj[i + m*S + n*M*S] = D[I_D[n + i*N] + m*K] * DtY / DtD[I_D[n + i*N] + I_D[n + i*N]*K]

        ### Signer
            signer[n + i*N] = math.copysign(1, DtY)

        ### LU decomposition
        # for i in range(0, S):
            for j in range(i, S):
                tmp = DtD[I_D[n + i*N] + I_D[n + j*N]*K]
                for k in range(0, i):
                    tmp -= L[k + j*S] * U[i + k*S]
                L[i + j*S] = tmp

            U[i + i*S] = 1
            for j in range(i+1, S):
                tmp = DtD[I_D[n + j*N] + I_D[n + i*N]*K] / L[i + i*S]
                for k in range(0, i):
                    tmp -= (L[k + i*S] * U[j + k*S]) / L[i + i*S]
                U[j + i*S] = tmp

        ### Solve system
        # for i in range(0, S):
            tmp = DtY / L[i + i*S]
            for j in range(0, i):
                tmp -= (L[j + i*S] * z[j + n*S]) / L[i + i*S]
            z[i + n*S] = tmp

        for i in range(S-1, 0-1, -1):
            tmp = z[i + n*S]
            for j in range(i+1, S):
                tmp -= U[j + i*S] * x[j + n*S]
            x[i + n*S] = tmp

        ### D@x
        for m in range(0, M):
           tmp = 0
           for s in range(0, S):
               tmp += D[I_D[n + s*N] + m*K] * x[s + n*S]
           matproj[m + n*M] = tmp

@cuda.jit
def k_updateD(D, I_D, Y, vecproj, matproj, signer):
    m = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if m < M:
        for n in range(0, N):
            for s in range(0, S):
                D[I_D[n + s*N] + m*K] += (Y[n + m*N] - matproj[m + n*M] + vecproj[s + m*S + n*M*S])*signer[n + s*N]


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
    
    M, N = smallSet.shape
    SS = S*S

    cuda.profile_start()
    t0 = time.time()
    D = ITKrM(smallSet, K, S, maxit)
    dt = time.time() - t0
    cuda.profile_stop()
    print("\n{}".format(dt))
    