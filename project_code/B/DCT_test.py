#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 08:46:27 2018

@author: niels
"""

def dct_picture(picture, sparsness, subpictures):
    import numpy as np
    import scipy as sp
    from scipy import fftpack
    import matplotlib.pyplot as plt
    from skimage.measure import compare_ssim
    
    DCT = fftpack.dct(fftpack.dct(picture.T, norm='ortho').T, norm='ortho')
    
    dct_allowance = sparsness * subpictures
    dct_allowance = int(np.ceil(np.sqrt(dct_allowance)))
    dct_copy = DCT.copy()
    dct_copy[dct_allowance:,:] = 0
    dct_copy[:,dct_allowance:] = 0
    
    IDCT = fftpack.idct(fftpack.idct(dct_copy.T, norm='ortho').T, norm='ortho')
    
    after = np.floor(IDCT)
    
    ssim = compare_ssim(picture, after)
#    print("DCT entries used: {}".format(dct_allowance**2))
#    print("DCT ssim: {}".format(ssim))
    
    return after, DCT, ssim, dct_allowance