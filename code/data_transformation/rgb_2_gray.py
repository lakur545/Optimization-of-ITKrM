# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 08:10:36 2018

@author: Niels
"""

from matplotlib import pyplot as plt
import my_funcs as N
import numpy as np

newfile=1          ## Flag to handle if one wants to load a new file         

if(newfile == 1):
    file_path=N.filePath()

dataAr=N.unpickle(file_path)  ## laod data (cifar-10)

info=dataAr[b'data']        ## save only the picture data to a variable

pic1=N.grayVec(info[4],32,32).reshape(32,32) ## take picture number 4 , put on gray scale, and reshape it to a 32x32 matrix, as that is the picture size


plt.imshow(pic1,plt.get_cmap('gray')) ## Plot the picture


infoG=np.array([N.grayVec(info[i],32,32) for i in range(info.shape[0])],dtype='uint8') ## convert all the picture data to gray scale

