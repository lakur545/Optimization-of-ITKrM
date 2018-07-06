# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 08:59:36 2018

@author: Niels
"""

#import numpy as np
#from timeit import default_timer as timer
#from numba import vectorize
#import numba.cuda as cuda
#
#
#
#
#@cuda.jit
#def Jit_Inverse(square_matrix):
#    import numpy as np
#    return np.linalg.inv(square_matrix)
#



#D_new = np.zeros((M,K))
#DtD = D_old.T@D_old
#for n in range(N):
#    DtY = D_old[:,I_D[:,n]].T @ Y[:,n]
#    matproj = np.repeat(np.array([ D_old[:,I_D[:,n]] @ np.linalg.inv(DtD[I_D[:,n,None], I_D[:,n]]) @ DtY ]).T, S, axis=1)
#    vecproj = D_old[:,I_D[:,n]] @ np.diag(np.diag( DtD[I_D[:,n,None], I_D[:,n]] )**-1*( DtY ))
#    signer = np.sign( DtY )
#    D_new[:,I_D[:,n]] = D_new[:,I_D[:,n]] + (np.repeat(Y[:,n,None], S, axis=1) - matproj + vecproj)*signer

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
