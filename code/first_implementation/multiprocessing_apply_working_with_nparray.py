# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 09:28:55 2018

@author: sebas
Playing around with the multiprocessing package 
"""

import numpy as np
import multiprocessing as mp
import time

def sleepy(x,k,l):
    time.sleep(0.5)
    y = x[1,1]

    return(x)

    

if __name__ == '__main__':
    M = mp.cpu_count() # Returns the number of hardware-supported threads
    #M = 4
    N = 4
    pool = mp.Pool(processes=M)
    t0 = time.time()
    case = [1,2,3,5]
    case2 = [3,4,5,6]
    d = np.random.rand(10,4) 
    start = 0
    stop = 2
    d_list = [d, start, stop] # To send it to apply_async it has to be packed as a list
    X = pool.apply_async(sleepy, (d_list)).get()
    dt = time.time() - t0
    pool.close()
    pool.join()
    
    print("Time to execute with {} core(s): {}".format(M,dt))
    print(d[0])
    print(X)
    
    

