# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""

import nibabel as nib
import matplotlib.pyplot as plt
from skimage.morphology import erosion, dilation, area_closing, square, remove_small_objects, h_maxima
import cv2
import scipy.ndimage as ndimage
import numpy as np

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data)) * 255


#Load the image
path = "C:\\temp\\TC_DATA\\Scoliose\\"#PATH to nifti files
file = '2preop.nii'
img = nib.load(path+file)
img_data = img.get_data()

print(img_data.shape)

# Screw reduction (for post-op images)
#img_data[img_data > 800] = 800

# Gaussian blurring
img_data = ndimage.gaussian_filter(img_data, sigma=(3, 3, 3), order=0)


#Slice selection
image = img_data[:,:,220]

#Regular image
plt.imshow(image,cmap='gray')
plt.show()


#Thresholding
thresh = 210
th_image = image > thresh
plt.imshow(th_image,cmap='gray')
plt.show()

#th_image = white_tophat(th_image, square(5))
#plt.imshow(th_image)
#plt.show()

#Sternum isolation
dl_image = ndimage.morphology.binary_fill_holes(th_image)

#dl_image = area_closing(th_image)
#Dilate to connect all bones of the sternum
#dl_image = dilation(dl_image,square(3))
#Remove ribs from image
dl_image = remove_small_objects(dl_image, min_size=1500)
#Undo the dilation
#dl_image = erosion(dl_image,square(3))


plt.imshow(dl_image,cmap='gray')
plt.show()



# Optional

#plt.imshow(dl_image_holefill, cmap='gray')
#plt.show()

plt.figure()
plt.imshow(image, cmap='gray')
plt.imshow(dl_image, cmap='jet', alpha=0.4)
plt.show()
#Find maximum locations
#print(h_maxima(dl_image, 100))
