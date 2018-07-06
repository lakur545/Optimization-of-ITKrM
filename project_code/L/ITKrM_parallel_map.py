# -*- coding: utf-8 -*-

import numpy as np
import N_timer
import multiprocessing as mp
import ImportImages
#import LoadFromDataBatch
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

def _parallel(threads, Y, D_old, I_D):
    pool = mp.Pool(processes=threads)
    M, N = Y.shape
    R = pool.map_async(_f, [(Y[:,n*N//4:(n+1)*N//4], K, S, maxit, D_old, I_D, n) for n in range(4)]).get()
    pool.close()
    pool.join()
    return R

def _f(d):
    Y, K, S, maxitr, D_old, I_D, i, DtD = d
    
    M, N = Y.shape
    D_new = np.zeros((M, K))
    for n in range(N):
        DtY = D_old[:,I_D[:,n]].T @ Y[:,n]
        matproj = np.repeat(np.array([ D_old[:,I_D[:,n]] @ np.linalg.inv(DtD[I_D[:,n,None], I_D[:,n]]) @ DtY ]).T, S, axis=1)
        vecproj = D_old[:,I_D[:,n]] @ np.diag(np.diag( DtD[I_D[:,n,None], I_D[:,n]] )**-1*( DtY ))
        signer = np.sign( DtY )
        D_new[:,I_D[:,n]] = D_new[:,I_D[:,n]] + (np.repeat(np.array([Y[:,n]]).T, S, axis=1) - matproj + vecproj)*signer
    return D_new


def itkrm(smallSet, K, S, maxit, nbr_threads, startD=np.array([1])):
    chunk_split = 4
    t0 = time.time()
    Y = smallSet
    M, N = Y.shape
    D_old = np.random.randn(M, K)
    I_D = np.zeros((S, N), dtype=np.int32)
    threads = nbr_threads
    pool = mp.Pool(processes = threads)
    t_par = 0
    for t in range(maxit):
        N_timer.Timer(t, maxit)
        for n in range(N):
            I_D[:,n] = max_atoms(D_old, Y[:,n], S)
        t_par_0 = time.time()
        DtD = D_old.T@D_old
        R = pool.map_async(_f, [(Y[:,n*N//(chunk_split*threads):(n+1)*N//(chunk_split*threads)], K, S, maxit, D_old, I_D[:,n*N//(chunk_split*threads):(n+1)*N//(chunk_split*threads)], n, DtD) for n in range((chunk_split*threads))]).get()
        t_par = t_par + time.time()-t_par_0
        D_new = np.sum(R, axis=0)
        #hugget fra Karin
        scale = np.sum(D_new*D_new, axis=0)
        iszero = np.where(scale < 0.00001)[0]
        D_new[:,iszero] = np.random.randn(M, len(iszero))
        #end hugget

        D_new = normalize_mat_col(D_new)
        D_old = 1*D_new
    t_seq = time.time()-t0-t_par
    dt = time.time() - t0
    print("\n{}".format(dt))
    print("t_seq: {}, t_par: {}, ratio: {}".format(t_seq, t_par, t_par/t_seq))
    print("Speed-up potential: {}".format(dt/t_seq))
    pool.close()
    pool.join()
    return D_new
    
    
    