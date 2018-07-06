# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 08:33:00 2018

@author: sebas, Niels, Jacob og Amalie
OMP as a function

# D has to be normalized when put into the function
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize # for normalizing dictionary

import N_timer

def OMP(D, y, e, sparsenesslevel, flag2, timer = 1):

    dv = np.asarray(D.shape)
    d = dv[0,]
    p = dv[1,]
    nv = np.asarray(y.shape)
    N = nv[1,]
    K = 0

    if flag2 == 1:
        e_or_S = sparsenesslevel
        K = sparsenesslevel
    else:
        e_or_S = p

    x_sparse_sorted = np.zeros((p,N))

    for n in range(N):
        if timer == 1:
            N_timer.Timer(n,N)
        r = y[:,n]*1 # initialize the residual
        S = np.zeros((d,p)) #The subset dictionary

        c = np.zeros((p,1))
        c[:,0] = p
        P = np.zeros((d,1))

        flag = 0

        for k in range(e_or_S):
            #print(k)
            if flag == 0:
                t_max = np.argmax(abs(D.T@r)) #step 2
                c[k,0] = t_max

                #print('k: {}. t_max: {}'.format(k, t_max))
    #
                for i in range(p):
                    if c[i,0] != p:
             #           print('i: {}'.format(i))
                        S[:,k] = D[:,c[i,0].astype(int)]
                        #print(S)

                P = S@np.linalg.pinv(S) # Step 3
                I = np.identity(d)
                r = (I-P)@y[:,n]

                if np.sum(abs(r)) < e and flag2 == 0:
                    K = k
                    flag = 1

        x_sparse = np.zeros((p,1))
        x_sparse = np.linalg.pinv(S)@y[:,n]

        #print('x_sparse: {}'.format(x_sparse))
        #print('n: {}'.format(n))


        for i in range(K):
            #print('i: {}'.format(i))
            x_sparse_sorted[c[i,0].astype(int),n] = x_sparse[i,]+1

        #print(x_sparse_sorted)
        #print('Sparsenes level {}:'.format(K))

    return x_sparse_sorted


if __name__ == "__main__":
    #Use the below code to test the OMP function
    import doctest

    #np.set_printoptions(threshold = np.nan)
    doctest.testfile("OMP_doctest.txt")

    np.random.seed(0)

    # For testing the OMP function
    M = 8; #number of rows in y
    L = 4; #number of coloumns in y (number of training sets)
    Q = 7; # Dimension of x
    D = np.random.randint(5, size = (M,Q))
    y = np.random.randint(5, size = (M,L))

    sparsenesslevel = 30
    error = 3
    flag = 1

    x_s = OMP(D,y, error, sparsenesslevel, flag)

    y_new = D@x_s

    print('\n\nsparseness level x: \n')
    for i in range(4):
        print(i,':', len(np.where(x_s[:,i]!=0)[0]))

    stepsize = 1
    image_x = np.arange(0,M,stepsize) #np.arange (start, (stop+stepsize), stepsize) returns nparray

    plt.figure('Before')
    plt.plot(image_x,y, 'o') # 'o' makes the dots
    plt.figure('After')
    plt.plot(image_x,y_new, 'o')
    plt.show()
