#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 14:25:23 2018

@author: niels
"""

import numpy as np
import cupy as cp

x_cpu = np.array([1, 2, 3])
l2_cpu = np.linalg.norm(x_cpu)

x_gpu = cp.array([1, 2, 3])
l2_gpu = cp.linalg.norm(x_gpu)
print("test")