#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 10:55:26 2018

@author: niels
"""
import numpy as np

Speed_up_wanted = 18
Sa = 17.5

M = ((Sa - 1) * Speed_up_wanted)/(Sa - Speed_up_wanted)

print(np.ceil(M))