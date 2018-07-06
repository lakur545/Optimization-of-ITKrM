# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 08:58:54 2018

@author: Jacob
"""

import numpy as np
import matplotlib.pyplot as plt

def ImportData(picType=-1, batchNumber=1):
    """
    Imports all pictures with the same label. 
    Parameters: 
        picType: Positive integer between [0:10]. -1 returns all pictures.
        batchNumber: Positive integer between [1:5]
    Returns:
        Dataset of pictures with size (number of pictures, N*N*3 pixels)
    """
    data = np.load('data_batch_{}'.format(batchNumber), encoding='bytes')
    picsAll = np.asarray(data[b'data'])
    if picType < 0 or picType > 9:
        return picsAll
    picIndex = np.where(np.asarray(data[b'labels'])==picType)[0]
    pics = picsAll[picIndex,:]
    lenpic = len(pics[0,:])//3
    return np.floor(0.299*pics[:, :lenpic]+0.587*pics[:, lenpic:2*lenpic]+0.114*pics[:, 2*lenpic:3*lenpic])

def ImportDataRGB(picType=-1, batchNumber=1):
    """
    Imports all pictures with the same label. 
    Parameters: 
        picType: Positive integer between [0:10]. -1 returns all pictures.
        batchNumber: Positive integer between [1:5]
    Returns:
        Dataset of pictures with size (number of pictures, N*N*3 pixels)
    """
    data = np.load('data_batch_{}'.format(batchNumber), encoding='bytes')
    picsAll = np.asarray(data[b'data'])
    if picType < 0 or picType > 9:
        return picsAll
    picIndex = np.where(np.asarray(data[b'labels'])==picType)[0]
    return picsAll[picIndex,:]

def PlotPicsRGB(pics, numberOfPics=-1):
    """
    Plots all pictures in a single plot.
    """
    nPics, nPixels = pics.shape
    if numberOfPics > 0 and numberOfPics < nPics:
        nPics = numberOfPics
    N = np.sqrt(nPixels/3).astype(int)
    sqrtNPics = np.ceil(np.sqrt(nPics)).astype(int)
    
    allPics = np.zeros((N*(sqrtNPics-np.argmin((sqrtNPics*(sqrtNPics-1), nPics-1))), N*sqrtNPics, 3), dtype=np.uint8)
    n = nPics
    for i in range(len(allPics[:,0])):
        for j in range(np.min((sqrtNPics, n))):
            allPics[i*N:(i+1)*N, j*N:(j+1)*N, :] = pics[i*sqrtNPics+j,:].reshape(3, N, N).transpose(1,2,0)
            n = n - 1
#    ax = plt.figure()
    plt.imshow(allPics, vmin=0, vmax=255)
#    return ax

def PlotPics(pics, numberOfPics=-1):
    nPics, nPixels = pics.shape
    if numberOfPics > 0 and numberOfPics < nPics:
        nPics = numberOfPics
    
    N = np.sqrt(nPixels).astype(int)
    sqrtNPics = np.ceil(np.sqrt(nPics)).astype(int)
    allPics = np.zeros((N*(sqrtNPics-np.argmin((sqrtNPics*(sqrtNPics-1), nPics-1))), N*sqrtNPics))
    n = nPics
    for i in range(allPics.shape[0]):
        for j in range(np.min((sqrtNPics, n))):
            allPics[i*N:(i+1)*N, j*N:(j+1)*N] = pics[i*sqrtNPics+j,:].reshape(N, N)
            n = n - 1
#    ax = plt.figure()
    plt.imshow(allPics, cmap='gray', vmin=0, vmax=255)
#    return ax

if __name__ == "__main__":
    picsRGB = ImportDataRGB(1)
    PlotPicsRGB(picsRGB)
    pics = ImportData(1)
    PlotPics(pics)

