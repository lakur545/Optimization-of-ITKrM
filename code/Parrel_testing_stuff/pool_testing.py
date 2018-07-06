# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 14:50:42 2018

@author: Niels
"""

from multiprocessing import Pool
from multiprocessing import Lock
import time

#def f(x):
#    for i in range(100):
#        for j in range(100):
#            for k in range(100):
#                x=x+0.0001
#    return x*x
#
#if __name__ == '__main__':
#    
#    
#    before=time.time()
#    with Pool(4) as p:
#        print(p.map(f, list(range(1000))))
#    pool_time = time.time() - before
#    
#    
#    before=time.time()
#    l=[]
#    for i in range(1000):
#        l+=[f(i)]
#    print(l)
#    seq_time = time.time() - before
#    print(pool_time)
#    print(seq_time)


#data_pairs = [ [3,np.array([3,4])], [4,[3,4]], [7,[3,4]], [1,[3,4]] ]
#
## define what to do with each data pair ( p=[3,5] ), example: calculate product
#def myfunc(p):
#    
#    return p[1]
#
#if __name__ == '__main__':
#    pool = multiprocessing.Pool(processes=4)
#    result_list = pool.map(myfunc, data_pairs)
#    print(result_list)
# 

def initializer():
    global data
    data = createObject()   


if __name__ == '__main__':
    pool = Pool(4, initializer, ())