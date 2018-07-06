# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:01:51 2018

@author: Niels
"""

import numpy as np

import N_timer


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

def itkrm(data,K,S,maxitr,startD=np.array([1])):
    M, N = data.shape
    if startD.all()==1:
        D_init = np.random.randn(M, K)
    else:
        D_init = startD
    Y = data
    I_D = np.zeros((S,N), dtype=np.int32)
    
    #Algorithm
    D_old = D_init
    for i in range(maxitr):
        start_time = N_timer.cont_timer(0,0)
        N_timer.Timer(i,maxitr)
        for n in range(N):
            I_D[:,n] = max_atoms(D_old,Y[:,n],S)
        D_new = np.zeros((M,K))
        DtD = D_old.T@D_old
        for n in range(N):
            DtY = D_old[:,I_D[:,n]].T @ Y[:,n]
            matproj = np.repeat(np.array([ D_old[:,I_D[:,n]] @ np.linalg.inv(DtD[I_D[:,n,None], I_D[:,n]]) @ DtY ]).T, S, axis=1)
            vecproj = D_old[:,I_D[:,n]] @ np.diag(np.diag( DtD[I_D[:,n,None], I_D[:,n]] )**-1*( DtY ))
            signer = np.sign( DtY )
            D_new[:,I_D[:,n]] = D_new[:,I_D[:,n]] + (np.repeat(Y[:,n,None], S, axis=1) - matproj + vecproj)*signer

    #Replace zero filled atoms
        scale = np.sum(D_new*D_new, axis=0)
        iszero = np.where(scale < 0.00001)[0]
        D_new[:,iszero] = np.random.randn(M, len(iszero))
    #

        D_new = normalize_mat_col(D_new)
        D_old = 1*D_new

    return D_old

if __name__ == "__main__":

