# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 08:33:00 2018

@author: sebas, Niels, Jacob og Amalie
OMP as a function

# D has to be normalized when put into the function
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import ImportImages as II
import N_timer


#@profile
def OMP(D, y, e, sparsenesslevel, flag2, timer=1):
    d, p = D.shape
    M, N = y.shape
    K = 0
    if flag2 == 1:
        e_or_S = sparsenesslevel
        K = sparsenesslevel
    else:
        e_or_S = p
    x_sparse_sorted = np.zeros((p, N))
    for n in range(N):
        if timer == 1:
            N_timer.Timer(n, N)
        r = y[:, n]*1   # initialize the residual
        S = np.zeros((d, p))    # The subset dictionary
        c = np.ones(p, dtype=np.int32)*p
        P = np.zeros(d)
        flag = 0
        for k in range(e_or_S):
            if flag == 0:
                t_max = np.argmax(abs(D.T@r))   # Step 2
                c[k] = t_max
                S[:, k] = D[:, c[k]]
                P = S[:, :k+1]@np.linalg.pinv(S[:, :k+1])   # Step 3
                I = np.identity(d)
                r = (I-P)@y[:, n]
                if np.sum(abs(r)) < e and flag2 == 0:
                    K = k
                    flag = 1
        x_sparse = np.zeros(p)
        x_sparse = np.linalg.pinv(S)@y[:, n]
        for i in range(K):
            x_sparse_sorted[c[i], n] = x_sparse[i]
    return x_sparse_sorted


if __name__ == "__main__":
    np.random.seed(0)
    file = h5py.File('dictionary', 'r')
    D = np.array(file['dictionary'])
    Y = np.array(file['testSet'])
    file.close()

    S = 100     # Sparsenesslevel
    E = 3   # Error
    e_or_s = 1

    x_s = OMP(D, Y, E, S, e_or_s)

    y_new = D@x_s

    W_data = 32
    H_data = 32
    N_subpic = 16

    beforeImage = II.MergeSmall(Y, W_data, H_data, N_subpic)
    afterImage = II.MergeSmall(y_new, W_data, H_data, N_subpic)

    print('\nsparseness level x:')
    for i in range(int((W_data*H_data)/(N_subpic*N_subpic))):
        print(i, ':', len(np.where(x_s[:, i] != 0)[0]))
    print('Average: {}'.format(len(np.where(x_s != 0)[0])/Y.shape[1]))

    pic_number = 0
    plt.figure('Before')
    plt.imshow(beforeImage[pic_number, :, :], cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.figure('After')
    plt.imshow(afterImage[pic_number, :, :], cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()
