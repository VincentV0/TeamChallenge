# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""

import nibabel as nib
import matplotlib.pyplot as plt
from skimage.morphology import erosion, dilation, area_closing, square, remove_small_objects, h_maxima

#Load the image
path = ""#PATH to nifti files
file = '1postop.nii'
img = nib.load(path+file)
img_data = img.get_data()

print(img_data.shape)

#Slice selection
image = img_data[:,:,220]

#Regular image
plt.imshow(image)
plt.show()


#Thresholding
thresh = 210
th_image = image > thresh
plt.imshow(th_image)
plt.show()

#th_image = white_tophat(th_image, square(5))
#plt.imshow(th_image)
#plt.show()

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

#Find maximum locations
#print(h_maxima(dl_image, 100))
