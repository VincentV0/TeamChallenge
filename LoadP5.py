# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import nibabel as nib
import matplotlib.pyplot as plt
from skimage.morphology import disk, erosion, dilation, area_closing, square, remove_small_objects, h_maxima
from skimage.measure import regionprops
import numpy as np

#Load the image
path = "" #Path to nifti files
file = '1postop.nii'
img = nib.load(path+file)
img_data = img.get_data()

print(img_data.shape)

#Slice selection
image = img_data[:,:,250]

#Regular image
plt.imshow(image)
plt.show()


#Thresholding
thresh = 210
th_image = image > thresh
plt.imshow(th_image)
plt.show()


#Sternum isolation
dl_image = area_closing(th_image)
#Dilate to connect all bones of the sternum
dl_image = dilation(dl_image,square(3))
#Remove ribs from image
dl_image = remove_small_objects(dl_image, min_size=1500)
#Undo the dilation
dl_image = erosion(dl_image,square(3))


plt.imshow(dl_image)
plt.show()

## Select center of mass for an image with just the sternum,
## to use as reference for images with multiple structures

labeled_foreground = dl_image.astype(int)
properties = regionprops(labeled_foreground)
center_of_mass = properties[0].centroid
print(center_of_mass)
##

## If image has multiple structures, use previous known location of sternum to 
## select sternum
def find_sternum(img, center_of_mass):
    nonzero = np.argwhere(img == True)
    distances = np.sqrt((nonzero[:,0] - center_of_mass[0]) ** 2 + (nonzero[:,1] - center_of_mass[1]) ** 2)
    nearest_index = np.argmin(distances)
    return nonzero[nearest_index]
print(find_sternum(dl_image, center_of_mass))
##

################ Fill the image to get rid of holes
dl_image = dilation(dl_image,disk(15))
dl_image = erosion(dl_image,disk(15))

plt.imshow(dl_image)
plt.show()

#################


#########################
def find_point5(image):
#Find location of point 5
#By using the processed image with only the sternum visible
    y=0
    x=0
    while y < image.shape[1]-1:
        for x in range(image.shape[0] - 1 ):
            if image[x,y] == True:
                point_5 = [x,y]
                break
            x=x+1
        y=y+1
    return point_5

point_5 = find_point5(dl_image)
print("Location point 5 = ", point_5)
