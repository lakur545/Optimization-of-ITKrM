# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 11:31:29 2018

@author: Niels
"""




import scipy.io as ios
import numpy as np

a=np.random.randn(10,10)
b="Hello world"
A={}
A['array']=a
A['string']=b


ios.savemat("test",A)