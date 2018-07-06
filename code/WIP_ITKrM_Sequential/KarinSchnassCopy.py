# -*- coding: utf-8 -*-
"""
Copy-paste of Karin Schnass implementation
"""
import numpy as np

import N_timer

def itkrm(data,K,S,maxit):
    # Setup of initial dictionary
    M, N = data.shape
    D_init = np.random.randn(M, K)
    Y = data
    
    # Algorithm
    D = D_init
    for t in range(maxit):
        N_timer.Timer(t,maxit)
        ip = D.T@Y
        absip = np.abs(ip)
        signip = np.sign(ip)
        
        # Finding the indicies of the "best" atoms
        I = np.argpartition(absip, -S, axis=0)[-S:]
        
        gram = D.T@D
        Dnew = np.zeros((M, K))
        
        # Calculating/training the atoms
        for n in range(N):
            res = Y[:,n]-D[:,I[:,n]]@np.linalg.pinv(gram[I[:,n].reshape(S,1),I[:,n]])@ip[I[:,n],n]
            Dnew[:,I[:,n]] = Dnew[:,I[:,n]] + np.real(np.outer(res, signip[I[:,n],n]))
            Dnew[:,I[:,n]] = Dnew[:,I[:,n]] + D[:,I[:,n]]@np.diag(absip[I[:,n],n])
      
        # Finding close to 0 atoms and replacing
        scale = np.sum(Dnew*Dnew, axis=0)
        iszero = np.where(scale < 0.00001)[0]
        Dnew[:,iszero] = np.random.randn(M, len(iszero))
        scale[iszero] = np.sum(Dnew[:,iszero]*Dnew[:,iszero], axis=0)
        
        # Normalize dictionary using prior found scales
        Dnew = Dnew@np.diag(1/np.sqrt(scale))
        D = Dnew*1
    return Dnew

