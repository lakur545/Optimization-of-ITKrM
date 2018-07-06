#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 10:21:01 2018

@author: niels
"""

import numpy as np
import cupy as cp
from timeit import default_timer as timer
from numba import vectorize
import matplotlib.pyplot as plt

@vectorize(['float32(float32)'], target='cuda')
def GPU_pow(a):
    return a ** a

def CPU_pow(a):
    return a ** a


size = [10**i for i in range(2)]
T_time = np.zeros((2, len(size)))
for i in range(len(size)):
    vec_size = size[i]
    
    a = np.array(np.random.sample(vec_size), dtype=np.float32)
    c = np.zeros(vec_size, dtype=np.float32)
    
    start = timer()
    c = GPU_pow(a)
    duration = timer() - start
    
    print("GPU time: {}".format(duration))
    T_time[1, i] = duration
    
    start = timer()
    c = cp.asarray(a) ** cp.asarray(a)
    duration = timer() - start
    print("Cupy time: {}".format(duration))    
    
    
    start = timer()
    c = CPU_pow(a)
    duration = timer() - start
    
    print("CPU time: {}".format(duration))
    T_time[0, i] = duration

plt.close('all')

plt.figure(4, dpi=200)

plt.title("Vector A Entries to the Power of Vector B Entries, CPU vs GPU")

plt.plot(size, 10**3*T_time[0, :], label="CPU time", linewidth=2.5)
plt.yscale("log")
plt.plot(size, 10**3*T_time[1, :], '--', label="GPU time, with transfer", linewidth=2.5)
plt.yscale("log")

plt.legend()    

plt.xlabel("Dimension [N]", fontsize=16)

plt.ylabel("Time [ms]", fontsize=16)

ax = plt.gca()
plt.setp(ax.get_xticklabels(), fontsize=14)
plt.setp(ax.get_yticklabels(), fontsize=14)
plt.grid(which="major",ls="-", color='grey')
plt.tight_layout()
plt.xscale("log")
plt.show()
