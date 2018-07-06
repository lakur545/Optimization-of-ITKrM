# -*- coding: utf-8 -*-
"""
Copy-paste of Karin Schnass implementation
"""

import numpy as np

import N_timer
from multiprocessing import Pool
from itertools import repeat

def All_k_atoms_4_n(M,K,S,Y,D,I,gram,ip,signip,absip,n):
    Dnew = np.zeros((M, K))
    res = Y[:,n]-D[:,I[:,n]]@np.linalg.pinv(gram[I[:,n].reshape(S,1),I[:,n]])@ip[I[:,n],n]
    Dnew[:,I[:,n]] = Dnew[:,I[:,n]] + np.real(np.outer(res, signip[I[:,n],n]))
    Dnew[:,I[:,n]] = Dnew[:,I[:,n]] + D[:,I[:,n]]@np.diag(absip[I[:,n],n])
    return Dnew




#@profile
def itkrm(data,K,S,maxit):
    M, N = data.shape
    D_init = np.random.randn(M, K)
    Y = data
    
    ### Algorithm
    D = D_init
    for t in range(maxit):
        N_timer.Timer(t,maxit)
#        print('Iteration: {}'.format(t+1))
        ip = D.T@Y
        absip = np.abs(ip)
        signip = np.sign(ip)
        I = np.argpartition(absip, -S, axis=0)[-S:]
        gram = D.T@D
        Dnew = np.zeros((M, K))
        for n in range(N):
            Dnew += All_k_atoms_4_n(M,K,S,Y,D,I,gram,ip,signip,absip,n)
#            res = Y[:,n]-D[:,I[:,n]]@np.linalg.pinv(gram[I[:,n].reshape(S,1),I[:,n]])@ip[I[:,n],n]
#            Dnew[:,I[:,n]] = Dnew[:,I[:,n]] + np.real(np.outer(res, signip[I[:,n],n]))
#            Dnew[:,I[:,n]] = Dnew[:,I[:,n]] + D[:,I[:,n]]@np.diag(absip[I[:,n],n])
#            if t==0 and n==0: print(Dnew[-1,:])
        scale = np.sum(Dnew*Dnew, axis=0)
        iszero = np.where(scale < 0.00001)[0]
#        print(len(iszero))
        Dnew[:,iszero] = np.random.randn(M, len(iszero))
        scale[iszero] = np.sum(Dnew[:,iszero]*Dnew[:,iszero], axis=0)
        Dnew = Dnew@np.diag(1/np.sqrt(scale))
        D = Dnew*1
    return Dnew

