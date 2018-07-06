# -*- coding: utf-8 -*-
"""
Copy-paste of Karin Schnass implementation
"""

import numpy as np

import N_timer


#@profile
def itkrm(data,K,S,maxit,startD):
    M, N = data.shape
#    if startD.all()==1:
#        print("haj")
##        D_init = np.random.randn(M, K)
#    else:
    D_init=startD[:,:K]
    Y = data
    pos=K
    ### Algorithm
    D = D_init
    for t in range(maxit):
        N_timer.Timer(t,maxit)
#        print('Iteration: {}'.format(t+1))
        ip = D.T@Y
        absip = np.abs(ip)
        signip = np.sign(ip)
        I = np.argsort(absip, axis=0)[-S:][::-1]
        gram = D.T@D
        Dnew = np.zeros((M, K))
        for n in range(N):
            res = Y[:,n]-D[:,I[:,n]]@np.linalg.pinv(gram[I[:,n].reshape(S,1),I[:,n]])@ip[I[:,n],n]
            Dnew[:,I[:,n]] = Dnew[:,I[:,n]] + np.real(np.outer(res, signip[I[:,n],n]))
            Dnew[:,I[:,n]] = Dnew[:,I[:,n]] + D[:,I[:,n]]@np.diag(absip[I[:,n],n])
        scale = np.sum(Dnew*Dnew, axis=0)
        iszero = np.where(scale < 0.00001)[0]
        newpos=pos+len(iszero)
        print(pos)
        print(len(iszero))
        print(newpos)
        Dnew[:,iszero] = startD[:,pos:newpos]
        pos=newpos
        scale[iszero] = np.sum(Dnew[:,iszero]*Dnew[:,iszero], axis=0)
        Dnew = Dnew@np.diag(1/np.sqrt(scale))
        D = 1*Dnew
#    print("\n")
#    print(pos)
#    print(res)
#    print(res.shape)
#    print(signip[I[:,N-1],N-1])
#    test=np.array([signip[I[:,N-1],N-1]])
#    test2=np.array([res])
#    print(test[:,::-1])
#    print(test2.T.shape)
#    print(Dnew[-1,:])
    return Dnew

