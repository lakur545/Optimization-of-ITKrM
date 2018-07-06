# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 13:57:18 2018

@author: sebas Import mat file
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

#batch_number = '09'

batch_number = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14']


for n in range(len(batch_number)):
    print(batch_number[n])
    if n == 0:
        img = nib.load('nifti_data/inplane0{}.nii'.format(batch_number[n]))
        data = np.array(img.dataobj)
    else:
        img = nib.load('nifti_data/inplane0{}.nii'.format(batch_number[n]))
        data_temp = np.array(img.dataobj)
        data = np.append(data, data_temp, axis = 2)
    print(data.shape)
#header = img.header
#print(header)

np.ndarray.tofile(data,'mri_scans')



#
#H = 128
#W = 128


pic_number  = 6

plt.figure('picture')
plt.imshow(data[:,:,pic_number])
plt.tight_layout()

pic_number  = 7

plt.figure('picture 2')
plt.imshow(data[:,:,pic_number])
plt.tight_layout()
plt.show()

print("done")
