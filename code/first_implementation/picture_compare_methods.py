# -*- coding: utf-8 -*-
"""
@author: Niels

This files includes some functions to calculate some different ways to evaluate the difference between two pictures
"""

import numpy as np

def dif_map(before,after):
    """
    
    This functions returns the difference between to pictures
    (normally an orignal and then a noisy/compressed version)
    
    """
    return before.astype('float64') - after.astype('float64')


def p_error(before,after):
    """
    
    This functions returns the p-value of  pixel error between two pictures
    (normally an orignal and then a noisy/compressed version)
    
    """
    return np.linalg.norm(dif_map(before,after),ord=2) / np.linalg.norm(before,ord=2)


def MSE(before,after):
    """
    
    This functions returns the Mean Square Error between two pictures
    (normally an orignal and then a noisy/compressed version)
    
    """
    heigth, width = before.shape
    return sum(sum((dif_map(before,after))**2)) / (heigth*width)



def PSNR(before,after,d_range=255):
    """
    
    This functions returns the Peak Signal to Noise between two pictures in dB
    (normally an orignal and then a noisy/compressed version)
    
    The value d_range is the dynamic range of the picture eg for a normal 8-bit grascale [0,255], gives a dynamic range of 255
    More generally it is ( 2^(# bits per pixel) ) - 1
    
    """
    return 10 * np.log10( (d_range**2) / (MSE(before,after)) )

def SSIM(before, after, window_size=1, d_range=255, k1=0.01, k2=0.03):
    """
    
    NOT YET DONE
    
    This function returns the Structural SIMilarity value between two pictures
    (normally an orignal and then a noisy/compressed version)
    
    The window_size is what integer value is needed to make the size of squares(N x N) that the pictures should be cut into and compared
    For the most part this will require that the input pictures are squares or divisble into squares
    
    The value d_range is the dynamic range of the picture eg for a normal 8-bit grascale [0,255], gives a dynamic range of 255
    More generally it is ( 2^(# bits per pixel) ) - 1
    
    k1, k2 are values that i do not know exactly what are, so probably keep the defaults.
    
    More info on Structural similarity can be found at
    https://en.wikipedia.org/wiki/Structural_similarity
    """
    heigth, width = before.shape
    
    if heigth%window_size!=0 or width%window_size!=0:
        print("This window_size will not work")
        return
    rows = heigth/window_size
    cols = width/window_size
    
    meanB = np.mean(before)
    meanA = np.mean(after)
    varB = np.var(before)
    varA = np.var(after)
    covBA = 0
    C1 = (k1 * d_range)**2
    C2 = (k2 * d_range)**2
    
    return
