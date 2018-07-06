# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 08:33:00 2018

@author: sebas
OMP as a function

# D has to be normalized when put into the function
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize # for normalizing dictionary
import N_timer

# For testing the OMP function 
#np.random.seed(0)
#M = 50; #number of rows in y
#L = 1000; #number of coloumns in y (number of training sets)
#Q = 100; # Dimension of x
#D = np.random.randint(5, size = (M,Q))
#y = np.random.randint(5, size = (M,L))

#@profile
def OMP(D, y, e):
    
    d, p = D.shape
    N = y.shape[1]
    K = 0
    x_sparse_sorted = np.zeros((p,N))
    I = np.identity(d)

    for n in range(N):
        N_timer.Timer(n,N)
        r = y[:,n]*1 # initialize the residual
        S = np.zeros((d,p)) #The subset dictionary

        c = np.zeros(p, dtype=np.int32)
        P = np.zeros(d)

        flag = 0
        for k in range(p):
            if flag == 0: 
                c[k] = np.argmax(abs(D.T@r)) #step 2
                S[:,k] = D[:,c[k]]
                P = S[:,:k+1]@np.linalg.pinv(S[:,:k+1]) # P = S[:,:k+1]@np.linalg.pinv(S[:,:k+1]) # Step 3
                r = (I-P)@y[:,n]
                if np.sum(abs(r)) < e:
                    K = k
                    flag = 1

        x_sparse = np.linalg.pinv(S)@y[:,n]
        x_sparse_sorted[c[:K+1],n] = x_sparse[:K+1]
#        for i in range(K+1):
#            x_sparse_sorted[c[i],n] = x_sparse[i]
    return x_sparse_sorted

    
#Use the below code to test the OMP function

#x_s = OMP(D,y, 1)
#y_new = D@x_s
#
#stepsize = 1
#image_x = np.arange(0,M,stepsize) #np.arange (start, (stop+stepsize), stepsize) returns nparray    
#    
#plt.figure(0)
#plt.plot(image_x,y) # 'o' makes the dots
#plt.plot(image_x,y_new,'o')
