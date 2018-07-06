# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:01:51 2018

@author: Niels
"""

import numpy as np

import N_timer

import multiprocessing as mp

# Set the number of threads for numpy.
# This is pretty usefull/necessary as it allows to run the apply async
# without numpy and apply fighting for cores.
import mkl
mkl.set_num_threads(1)

#


# split_data is not as such a part of the itkrm algorithm
def split_data(M, nSamples):  # For now it only works if nSamples/M = int
    # print("went in split_data")
    split = [None]*M
    for m in range(M):
        split[m] = int(nSamples/M*(m))

    return split


def max_atoms(D, yn, S):
    All_Abs = abs(D.T @ yn)
    indexs = np.argpartition(All_Abs, -S)[-S:]
    return indexs


def proj(vector_space):
    try:
        vector_space.shape[1]
    except:
        vector_space = np.array([vector_space]).T
    return vector_space @ np.linalg.pinv(vector_space)


def normalize_vec(vector):
    import numpy as np
    return vector / np.linalg.norm(vector, ord=2)


def normalize_mat_col(matrix):
    return np.array([normalize_vec(matrix[:, n]) for n in range(matrix.shape[1])]).T


def inner_loop(D_old, D_new, I_D, Y, S, q):
    # print("Proces and 1st entry Y: {} \n".format(Y[1,1]))
    M, N = Y.shape
    for n in range(N):
        # print("Proces {}, iteration {}, entry Y: {} \n".format(q,n,Y[1,n]))
        matproj = np.repeat(np.array([proj(D_old[:, I_D[:, n]]) @ Y[:, n]]).T, S, axis=1)
        vecproj = D_old[:, I_D[:, n]] @ np.diag(np.diag(D_old[:, I_D[:, n]].T @ D_old[:, I_D[:, n]])**-1*(D_old[:, I_D[:, n]].T @ Y[:, n]))
        signer = np.sign(D_old[:, I_D[:, n]].T @ Y[:, n])
        D_new[:, I_D[:, n]] = D_new[:, I_D[:, n]] + (np.repeat(np.array([Y[:, n]]).T, S, axis=1) - matproj + vecproj) * signer
    return D_new


#@profile
def itkrm(data, K, S, maxitr, startD=np.array([1])):
        # Setting up the pool for parallel processing using map:
    nCores = mp.cpu_count()
#    nCores = nCores
    pool = mp.Pool(processes=nCores)
    qd = nCores  # how many parts the training data is divided into

    [rows, coloumns] = data.shape
    print("nTrainingData: {}\n".format(coloumns))

    M, N = data.shape
    if startD.all() == 1:
        D_init = np.random.randn(M, K)
    else:
        D_init = startD
    Y = data
    I_D = np.zeros((S, N), dtype=np.int32)
#    N_timer.log(0,log_s='20 data test, 14/03',open_file=1)
    # Algorithm
    D_old = D_init
    for i in range(maxitr):
        start_time = N_timer.cont_timer(0, 0)
        N_timer.Timer(i, maxitr)
        for n in range(N):
            I_D[:, n] = max_atoms(D_old, Y[:, n], S)
        D_new = np.zeros((M, K))
        # for n in range(int(N/qd)):
        # print(n)
#            matproj = np.repeat(np.array([ proj(D_old[:,I_D[:,n]])@Y[:,n] ]).T, S, axis=1)
#            vecproj = D_old[:,I_D[:,n]] @ np.diag(np.diag(D_old[:,I_D[:,n]].T @ D_old[:,I_D[:,n]] )**-1*(D_old[:,I_D[:,n]].T@Y[:,n]))
#            signer = np.sign(D_old[:,I_D[:,n]].T@Y[:,n])
#            D_new[:,I_D[:,n]] = D_new[:,I_D[:,n]] + (np.repeat(np.array([Y[:,n]]).T, S, axis=1) - matproj + vecproj)*signer
#
        # ------------ the above as a function ---------------------------------
        split_q = split_data(qd, coloumns)
        D_new_split = [pool.apply_async(inner_loop, ((D_old, D_new, I_D[:, q:int(q + coloumns/qd):1], Y[:, q:int(q + coloumns/qd):1], S, q))) for q in split_q]
        # print("Left dictionary_split")
        D_new_temp = np.array([D_new_split[q].get() for q in range(len(split_q))])
        # print("dict_temp: {}".format(D_new_temp.shape))

        # print("D_new_temp: {}".format(D_new_temp[1,:,:]))

        # Summing the contributions from each CPU
        for q in range(qd):
            # print("q: {}".format(q))
            D_new = np.add(D_new, np.reshape(D_new_temp[q, :, :], (M, K)))

#        for n in range(N):
#            D_new = inner_loop(D_old,D_new, I_D, Y, S)

        # ----------------------------------------------------------------------

#            for k in I_D[:,n]:
#                vecproj = D_old[:,k] * (D_old[:,k].T@D_old[:,k])**-1 * (D_old[:,k].T@Y[:,n])
#                signer = np.sign(D_old[:,I_D[:,n]].T@Y[:,n])
#                D_new[:,k] = D_new[:,k]+(Y[:,n]-matproj+vecproj[:,m])*signer
    # hugget fra Karin
        scale = np.sum(D_new*D_new, axis=0)
        iszero = np.where(scale < 0.00001)[0]
        D_new[:, iszero] = np.random.randn(M, len(iszero))
    # end hugget

        D_new = normalize_mat_col(D_new)
        D_old = 1*D_new
#        N_timer.log(N_timer.cont_timer(start_time,1))
#    N_timer.log("end",open_file=-1)
    return D_old


if __name__ == "__main__":
    pass
