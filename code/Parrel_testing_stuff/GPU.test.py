# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 10:19:40 2018

@author: Niels
"""

import numpy as np
from timeit import default_timer as timer
from numba import vectorize
import numba.cuda as cuda



@cuda.jit
def JVectorAdd(a, b, c):
    x = cuda.grid(1)
    if x < c.size:
        c[x] = a[x] + b[x]

@vectorize(["float32(float32, float32)"], target='cuda')
def VectorAdd(a,b):
    return a+b

def SVectorAdd(a,b):
    return a+b

def main():
    VectorAdd(1,1)
       
    N = 32000000
    
    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)
    C = np.zeros(N, dtype=np.float32)
    D = np.zeros(N, dtype=np.float32)
    E = np.zeros(N, dtype=np.float32)
    JVectorAdd(A,B,E) 
    E = np.zeros(N, dtype=np.float32)
    
    start = timer()
    C = VectorAdd(A,B)
    vectoradd_time = timer() - start
    
    print("GPU Vectorized time: {}".format(vectoradd_time))
    
    start = timer()
    D = SVectorAdd(A,B)
    vectoradd_time = timer() - start
    
    print("CPU time: {}".format(vectoradd_time))
    
    start = timer()
    JVectorAdd(A,B,E)
    vectoradd_time = timer() - start
    
    print("GPU jit time: {}".format(vectoradd_time))
    
    print("Are the results the same: {}".format(np.allclose(C,D,E)))
    
if __name__ == '__main__':
    main()