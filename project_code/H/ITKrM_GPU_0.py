# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:01:51 2018

@author: Niels
"""

import numpy as np

import cupy as cp

import N_timer

cp.random.seed(0)

def max_atoms(D,yn,S):
    All_Abs = cp.abs(D.T @ yn)
    I = cp.argpartition(All_Abs, -S)[-S:]
    return I

def normalize_vec(vector):
    return vector / cp.linalg.norm(vector,ord=2)

def normalize_mat_col(matrix):
    return cp.vstack([normalize_vec(matrix[:,n]) for n in range(matrix.shape[1])]).T

#@profile
def itkrm(data,K,S,maxitr,startD=np.array([1])):
    M, N = data.shape
    if startD.all()==1:
        D_init = np.random.randn(M, K)
    else:
        D_init = startD
        
    #Algorithm    
    GPU_D_old = cp.asarray(D_init)
       
    GPU_Y = cp.asarray(data)
    
    GPU_M = int(cp.asarray(M))
    
    GPU_N = int(cp.asarray(N))
    
    GPU_S = int(cp.asarray(S))
    
    GPU_maxitr = int(cp.asarray(maxitr))
    
    GPU_I_D = cp.zeros((S,N),dtype=cp.int32)
   
    for i in range(GPU_maxitr):
        start_time = N_timer.cont_timer(0,0)
        N_timer.Timer(i,maxitr)
        for n in range(GPU_N):
            GPU_I_D[:,n] = max_atoms(GPU_D_old,GPU_Y[:,n],GPU_S)

        GPU_D_new = cp.zeros((M,K))
        
        GPU_DtD = GPU_D_old.T @ GPU_D_old

        for n in range(GPU_N):
            GPU_DtY = GPU_D_old[:,GPU_I_D[:,n]].T @ GPU_Y[:,n]
            GPU_matproj = cp.repeat((GPU_D_old[:,GPU_I_D[:,n]] @ cp.linalg.inv(GPU_DtD[GPU_I_D[:,n,None], GPU_I_D[:,n]]) @ GPU_DtY)[:,None],GPU_S,axis=1)
            GPU_vecproj = GPU_D_old[:,GPU_I_D[:,n]] @ cp.diag(cp.diag( GPU_DtD[GPU_I_D[:,n,None], GPU_I_D[:,n]] )**-1*( GPU_DtY ))
            GPU_signer = cp.sign( GPU_DtY )
            GPU_D_new[:,GPU_I_D[:,n]] = GPU_D_new[:,GPU_I_D[:,n]] + (cp.repeat(GPU_Y[:,n,None], S, axis=1) - GPU_matproj + GPU_vecproj)*GPU_signer
            
            


        GPU_scale = cp.sum(GPU_D_new*GPU_D_new, axis=0)
        GPU_iszero = cp.where(GPU_scale < 0.00001)[0]
#        GPU_D_new[:,GPU_iszero] = np.random.randn(GPU_M, len(GPU_iszero))  # generate random with GPU
        GPU_D_new[:,GPU_iszero] = cp.asarray(np.random.randn(M, len(GPU_iszero)))  # generate random with CPU
    #end hugget
        GPU_D_new = normalize_mat_col(GPU_D_new)
        GPU_D_old = 1*GPU_D_new
    return cp.asnumpy(GPU_D_old)

if __name__ == "__main__":

    Y=Y_fake_dic
    itr=20

    atoms=100

    sparsness=5

    D=itkrm_niels(Y,atoms,sparsness,itr)

    X=OMP(D,Y,1)
