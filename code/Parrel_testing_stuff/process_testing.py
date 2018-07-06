# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 08:51:35 2018

@author: Niels
"""
import multiprocessing as mp
import time

def init(aa, vv):
    global a, v
    a = aa
    v = vv

def worker(i):
    time.sleep(1)
    v.value +=1

if __name__ == "__main__":
    N = 10
    a = mp.Array('i', [0]*N)
    v = mp.Value('i', 3)
    p = mp.Pool(4, initializer=init, initargs=(a, v))
    p.map(worker, range(N))
    print(a[:])
    print(v.value)