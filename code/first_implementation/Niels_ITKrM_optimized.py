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


#@profile
def itkrm(data,K,S,maxitr,startD=np.array([1])):

    M, N = data.shape
    if startD.all()==1:
        D_init = np.random.randn(M, K)
    else:
        D_init=startD
    Y = data
    I_D=np.zeros((S,N), dtype=np.int32)

    #Algorithm
    D_old=D_init
    #signer = Y.T@Y
    for i in range(maxitr):
        N_timer.Timer(i,maxitr)
        signer=np.sign(D_old.T@Y)
        for n in range(N):
            I_D[:,n]=max_atoms(D_old,Y[:,n],S)
        D_new=np.zeros((M,K))
        for n in range(N):
            for k in range(K):
                indicator=np.any(I_D[:,n]==k)
                if indicator==1:
                    matproj=proj(D_old[:,I_D[:,n]])@Y[:,n]
                    vecproj=proj(D_old[:,k])@Y[:,n]
                    D_new[:,k]=D_new[:,k]+(Y[:,n]-matproj+vecproj)*signer[k,n]
            if i==0 and n==0: print(D_new[-1,:])
    #hugget fra Karin
        scale = np.sum(D_new*D_new, axis=0)
        iszero = np.where(scale < 0.00001)[0]
        print(len(iszero))
        D_new[:,iszero] = np.random.randn(M, len(iszero))
    #end hugget

        D_new = normalize_mat_col(D_new)
        D_old = 1*D_new
    return D_old

if __name__ == "__main__":

    Y=Y_fake_dic
    itr=20

    atoms=100

    sparsness=5

    D=itkrm_niels(Y,atoms,sparsness,itr)

    X=OMP(D,Y,1)
