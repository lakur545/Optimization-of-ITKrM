# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:01:51 2018

@author: Niels
"""

import numpy as np


def Timer(itr=1,i=[0],t=[0]):
    import time
    itr=itr-1    
    if i[0]==1:
        timed=time.time()-t[0]                                              # calculate an estimate of total run time before done
        mins=int((itr*timed)/60)
        secs=int((itr*timed)%60)
        print("Estimate until done: {} minutes {} seconds".format(mins,secs))
        i[0]=0
        t[0]=0
    else: 
        t[0]+=time.time()
        i[0]+=1 # mutable variable get evaluated ONCE
    return 


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



def itksm(data,K,S,maxitr):
    M, N = data.shape
    D_init = np.random.randn(M, K)
    Y = data
    I_D=np.zeros((S,N), dtype=np.int32)
    
    #Algorithm
    D_old=D_init
    for i in range(maxitr):
        if i<2:Timer(maxitr)
        print('Iteration: {}'.format(i+1))
        for n in range(N):
            I_D[:,n]=max_atoms(D_old,Y[:,n],S)
        D_new=np.zeros((M,K))    
        for k in range(K):
#            print(k)
            for n in range(N):
                matproj=proj(D_old[:,I_D[:,n]])@Y[:,n]
                vecproj=proj(D_old[:,k])@Y[:,n]
                signer=np.sign(Y[:,n].T@Y[:,n])
                indicator=np.any(I_D[:,n]==k)
    #            print(matproj)
    #            print(vecproj)
    #            print(signer)
    #            print(indicator)
                D_new[:,k]=D_new[:,k]+Y[:,n]*signer*indicator
    #            print(D_new[:,k])
                
    #hugget fra Karin
        scale = np.sum(D_new*D_new, axis=0)
        iszero = np.where(scale < 0.00001)[0]
        D_new[:,iszero] = np.random.randn(M, len(iszero))
    #end hugget  
       
        D_old=normalize_mat_col(D_new) 
    return D_old   
    
if __name__ == "__main__":
    
    Y=Y_fake_dic
    itr=20
    
    atoms=100
    
    sparsness=5   
    
    D=itkrm_niels(Y,atoms,sparsness,itr)
    
    X=OMP(D,Y,1)