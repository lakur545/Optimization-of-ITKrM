# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 12:47:49 2018

@author: Niels
"""
import math
import os
from time import time
import numpy as np
import pycuda.autoinit
import scipy as sp
import pycuda.gpuarray as gpuarray
import skcuda.linalg as linalg
from numba import cuda
if os.system("cl.exe"):
    os.environ['PATH'] += ';'+r"C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin"
if os.system("cl.exe"):
    raise RuntimeError("cl.exe still not found, path probably incorrect")

@cuda.jit
def GPU_pin(A,B):
    B = np.linalg.pinv(A)
    return B


to_proj = A.T
To_proj = proj(to_proj)
X_GPU = gpuarray.to_gpu(to_proj.T)
Z_GPU = linalg.pinv(X_GPU, lib='cusolver')
to_proj = to_proj @ Z_GPU.get().T
