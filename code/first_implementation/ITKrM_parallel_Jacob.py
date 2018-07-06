# -*- coding: utf-8 -*-

import numpy as np
import N_timer
import multiprocessing as mp
import ImportImages
import LoadFromDataBatch
import os
import time

import mkl
mkl.set_num_threads(1)

def max_atoms(D,yn,S):
    All_Abs = abs(D.T @ yn)
    I = np.argpartition(All_Abs, -S)[-S:]
    return I

def proj(vector_space):
    try:
        vector_space.shape[1]
    except:
        vector_space = np.array([vector_space]).T
    return vector_space @ np.linalg.pinv(vector_space)

def normalize_vec(vector):
    import numpy as np
    return vector / np.linalg.norm(vector,ord=2)

def normalize_mat_col(matrix):
    return np.array([normalize_vec(matrix[:,n]) for n in range(matrix.shape[1])]).T

np.random.seed(0)
K = 200
S = 40
maxit = 20
N = 1024

W_data = 32
H_data = 32
N_subpic = 16

data = LoadFromDataBatch.ImportData(7, 1)
test_data = LoadFromDataBatch.ImportData(7, 'test')

data = data[:600,:]

smallSet = ImportImages.ExtractSmall(data.T, W_data, H_data, N_subpic)

def _parallel(threads, Y, D_old, I_D):
    pool = mp.Pool(processes=threads)
    M, N = Y.shape
    R = pool.map_async(_f, [(Y[:,n*N//4:(n+1)*N//4], K, S, maxit, D_old, I_D, n) for n in range(4)]).get()
    pool.close()
    pool.join()
    return R

def _f(d):
    Y, K, S, maxitr, D_old, I_D, i = d
    #time.sleep(float(i)/10.)
    
    M, N = Y.shape
    D_new = np.zeros((M, K))
    for n in range(N):
        matproj = np.repeat(np.array([ proj(D_old[:,I_D[:,n]])@Y[:,n] ]).T, S, axis=1)
        vecproj = D_old[:,I_D[:,n]] @ np.diag(np.diag(D_old[:,I_D[:,n]].T @ D_old[:,I_D[:,n]] )**-1*(D_old[:,I_D[:,n]].T@Y[:,n]))
        signer = np.sign(D_old[:,I_D[:,n]].T@Y[:,n])
        D_new[:,I_D[:,n]] = D_new[:,I_D[:,n]] + (np.repeat(np.array([Y[:,n]]).T, S, axis=1) - matproj + vecproj)*signer
#    pid = os.getpid()
#    print("{}, {}".format(i, pid))
    return D_new

if __name__ == "__main__":
    Y = smallSet
    M, N = Y.shape
    D_old = np.random.randn(M, K)
    I_D = np.zeros((S, N), dtype=np.int32)
    maxit = 20
    pool = mp.Pool(processes = 4)
    t0 = time.time()
    for t in range(maxit):
#        start_time = N_timer.cont_timer(0, 0)
        N_timer.Timer(t, maxit)
        for n in range(N):
            I_D[:,n] = max_atoms(D_old, Y[:,n], S)
        R = pool.map_async(_f, [(Y[:,n*N//4:(n+1)*N//4], K, S, maxit, D_old, I_D[:,n*N//4:(n+1)*N//4], n) for n in range(4)]).get()
        D_new = np.sum(R, axis=0)
        #hugget fra Karin
        scale = np.sum(D_new*D_new, axis=0)
        iszero = np.where(scale < 0.00001)[0]
        D_new[:,iszero] = np.random.randn(M, len(iszero))
        #end hugget

        D_new = normalize_mat_col(D_new)
        D_old = 1*D_new
#        print(D_new)
    dt = time.time() - t0
    print("\n{}".format(dt))
    print(D_new)
    pool.close()
    pool.join()
    
    
    