# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 13:52:51 2018

@author: sebas
"""

import numpy as np

M = 10000
s_a = 1/62.98
limit = 10
for m in range(M):
    S_a = m/((m-1)*s_a+1)
    if S_a > limit:
        print("m: {}, S_a: {}".format(m, S_a))
        limit = limit+10