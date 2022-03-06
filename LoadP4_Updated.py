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
import time
from ScrollPlot import ScrollPlot

def NormalizeData(data):
    """ Normalizes data to the [0, 255] range; not required right now """
    return (data - np.min(data)) / (np.max(data) - np.min(data)) * 255

# Start timer
start_time = time.time()

# Load the image
path = "C:\\temp\\TC_DATA\\Scoliose\\" # PATH to nifti files
file = '9preop.nii'
img = nib.load(path+file)
img_data = img.get_fdata()

# Set parameters
MIN_THRESHOLD_BONE = 210
MAX_THRESHOLD_ANTI_METAL = 800 # for post-op images only
MIN_THRESHOLD_STRUCTURE_SIZE = 1500

# Screw reduction (for post-op images only)
if file[1:7] == 'postop':
    img_data[img_data > MAX_THRESHOLD_ANTI_METAL] = MAX_THRESHOLD_ANTI_METAL

# Gaussian blurring
img_data_blurred = ndimage.gaussian_filter(img_data, sigma=(3, 3, 3), order=0)

# Create empty arrays to store data in
thresholded_3d = np.zeros(img_data.shape)
dl_image_3d = np.zeros(img_data.shape)

for slice in range(img_data.shape[2]):
    # Slice selection
    image = img_data_blurred[:,:,slice]

    # Thresholding
    th_image = image > MIN_THRESHOLD_BONE

    # Spine isolation
    dl_image = ndimage.morphology.binary_fill_holes(th_image)
    dl_image = remove_small_objects(dl_image, min_size=MIN_THRESHOLD_STRUCTURE_SIZE)

    # Store 3D data
    thresholded_3d[:,:,slice] = th_image
    dl_image_3d[:,:,slice] = dl_image

print("Processing the image took %s seconds." % (time.time() - start_time))
#print("To visualize a single slice, use 'python -i LoadP4.3D' and type 'visuals(slice_nr)' when the processing is done!")

def visuals(slice_nr):
    ### VISUALIZATION
    #Regular image
    plt.imshow(img_data[:,:,slice_nr],cmap='gray')
    plt.show()

    # Thresholded image
    plt.imshow(thresholded_3d[:,:,slice_nr],cmap='gray')
    plt.show()

    # Image with holes filled and spine isolated
    plt.imshow(dl_image_3d[:,:,slice_nr],cmap='gray')
    plt.show()

    plt.figure()
    plt.imshow(img_data[:,:,slice_nr], cmap='gray')
    plt.imshow(dl_image_3d[:,:,slice_nr], cmap='jet', alpha=0.4)
    plt.show()


    #Find maximum locations
    #print(h_maxima(dl_image, 100))

# Scroll_through image
fig, ax = plt.subplots(1, 1)
sp = ScrollPlot(ax, img_data, dl_image_3d)
fig.canvas.mpl_connect('scroll_event', sp.on_scroll)
plt.show()