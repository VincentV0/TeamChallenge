# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""

import nibabel as nib
import matplotlib.pyplot as plt
from skimage.morphology import erosion, dilation, area_closing, square, remove_small_objects, h_maxima
import scipy.ndimage as ndimage
import numpy as np
from ScrollPlot import ScrollPlot
import time

def NormalizeData(data):
    """ Normalizes data to the [0, 255] range; not required right now """
    return (data - np.min(data)) / (np.max(data) - np.min(data)) * 255

# Timer
start_time = time.time()

# Load the image
path = "C:\\temp\\TC_DATA\\Scoliose\\" # PATH to nifti files
file = '2preop.nii'
img = nib.load(path+file)
img_data = img.get_fdata()

# Set parameters
MIN_THRESHOLD_BONE = 210
MAX_THRESHOLD_ANTI_METAL = 800 # for post-op images only
MIN_THRESHOLD_STRUCTURE_SIZE = 8000000 # determine through trail and error


# Screw reduction (for post-op images only)
if file[1:7] == 'postop':
    img_data[img_data > MAX_THRESHOLD_ANTI_METAL] = MAX_THRESHOLD_ANTI_METAL

# Gaussian blurring
img_data_blurred = ndimage.gaussian_filter(img_data, sigma=(3, 3, 3), order=0)
print("Step 1 (Image blurring) complete")

# Thresholding
th_image = img_data_blurred > MIN_THRESHOLD_BONE
print("Step 2 (Image thresholding) complete")

dl_image = ndimage.morphology.binary_fill_holes(th_image)
print("Step 3 (Filling holes) complete")
#dl_image = remove_small_objects(dl_image, min_size=MIN_THRESHOLD_STRUCTURE_SIZE)
#print("Step 4 (Removing small objects) complete")


print("Processing the image took %s seconds." % (time.time() - start_time))
print("To visualize a single slice, use 'ipython -i LoadP4.3D' and type 'visuals(slice_nr)' when the processing is done!")

def visuals(slice_nr):
    ### VISUALIZATION
    #Regular image
    plt.imshow(img_data[:,:,slice_nr],cmap='gray')
    plt.show()

    # Thresholded image
    plt.imshow(th_image[:,:,slice_nr],cmap='gray')
    plt.show()

    # Image with holes filled and spine isolated
    plt.imshow(dl_image[:,:,slice_nr],cmap='gray')
    plt.show()

    plt.figure()
    plt.imshow(img_data[:,:,slice_nr], cmap='gray')
    plt.imshow(dl_image[:,:,slice_nr], cmap='jet', alpha=0.4)
    plt.show()

    #Find maximum locations
    #print(h_maxima(dl_image, 100))

# Scroll_through image
fig, ax = plt.subplots(1, 1)
sp = ScrollPlot(ax, img_data, dl_image)
fig.canvas.mpl_connect('scroll_event', sp.on_scroll)
plt.show()