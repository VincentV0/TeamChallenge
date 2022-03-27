"""
Obtaining thoracic landmarks and parameters from CT image.

Project of Team Challenge 2022 - Group 4.

Last updated: 21 Mar 2022
"""

# Import general libraries
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import math
from copy import deepcopy

# Import specific code
from readData      import *
from LM1368_utils  import load_LM1368
from LM2_utils     import load_LM2
from LM57_utils    import load_LM57
from utils         import rotate, rotate_landmarks, sagittal_diameter, haller_index
from interpol_outl import interpolate, find_outliers
from ScrollPlot    import ScrollPlot
from interpol_alternative import interpol_alt

# Set the path where all data can be found.
# Data is assumed to have the following structure (with ? being a number):
# DATA_PATH/
#    ?postop.nii
#    ?postop_mask1.nii
#    ?postop_mask2.nii
#    ?postop.xml
#    ?preop.nii
#    ?preop_mask1.nii
#    ?preop_mask2.nii
#    ?preop.xml

DATA_PATH = r"C:\\temp\\TC_DATA\\Scoliose"

# Ask the user for the image ID. As described above, this should be something like "1postop"
print("Please enter the image ID that should be processed: ")
image_ID = input('> ')
if image_ID[1:] == 'postop': postop = True
else: postop = False

# Load the .nii and .xml files
nifty_path = os.path.join(DATA_PATH, image_ID + ".nii")
xml_path   = os.path.join(DATA_PATH, image_ID + ".xml")
mask1_path = os.path.join(DATA_PATH, image_ID + "_mask1.nii")
mask2_path = os.path.join(DATA_PATH, image_ID + "_mask2.nii")

print('\nLoading images and manual annotations (for validation)...')
img            = read_nii(nifty_path, rotate=True)
img_no_rotate  = read_nii(nifty_path, rotate=False)
true_markers   = read_xml(xml_path, rotate=True)
mask1, mask2   = read_nii_masks(mask1_path, mask2_path, rotate=True)

# Setting some variables:
image_origin = (img.shape[0]//2, img.shape[1]//2)
landmarks = dict()
#...

# Rotate the true markers 
true_markers = rotate_landmarks(true_markers, image_origin)

### PART 1 - Landmarks 1, 3, 6 and 8; lung contours; surface 1
print('\nStarting part 1: Loading landmarks 1, 3, 6 and 8, lung contours and surface 1...')
landmarks[1], landmarks[3], landmarks[6], landmarks[8], \
    surface1, lung_segmentation, empty_slices = load_LM1368(img)
print('Part 1 finished.')

### PART 2 - Landmark 2
print('\nStarting part 2: Loading landmark 2')
landmarks[2] = load_LM2(img, lung_segmentation)
landmarks[2] = rotate_landmarks(landmarks[2], image_origin,math.pi/2)

print('Part 2 finished.')

### PART 3 - Landmarks 5 and 7 (4 yet to implement here)
print('\nStarting part 3: Loading landmarks 5 and 7')
#landmarks[5], landmarks[7], dl_image = load_LM57(img_no_rotate, postop)

# Rotate points 5 and 7 to new frame of reference.
#landmarks[5] = rotate_landmarks(landmarks[5], image_origin)
#landmarks[7] = rotate_landmarks(landmarks[7], image_origin)
print('Part 3 finished.')

# Make ScrollPlot
fig1, ax1 = plt.subplots()
sp1 = ScrollPlot(ax1, img, lung_segmentation, landmarks, true_markers)
fig1.canvas.mpl_connect('scroll_event', sp1.on_scroll)
fig1.canvas.mpl_connect('button_press_event', sp1.on_click)
plt.show()
point_coords = sp1.get_marked_points()

# Print selected point coordinates
if point_coords.size > 0:
    print("Selected points:")
    print("{:10} {:10} {:10}".format('x','y','z'))
    for x,y,z in point_coords:
        print("{:10} {:10} {:10}".format(x,y,z))

#for lm in landmarks:
#    after_interpolation=interpolate(landmarks[lm][:,0], landmarks[lm][:,1])
#    landmarks[lm][:,0], landmarks[lm][:,1], inds=find_outliers(after_interpolation[0], after_interpolation[1])
landmarks2 = interpol_alt(deepcopy(landmarks))

# TODO here:
# - Implement translation between preop and postop to translate points
# - Inter/Extrapolate points (because some are quite poor)
# - Validation (both using the given landmarks and our eyes)
# --- how to discard points when they are not good enough
# ...

fig2, ax2 = plt.subplots()
for lm in landmarks:
    x = landmarks[lm][:,0]
    y = landmarks[lm][:,1]
    ax2.plot(x, label=f'x_{lm}')
    ax2.plot(y, label=f'y_{lm}')
    ax2.legend()
plt.show()

fig3, ax3 = plt.subplots()
for lm in landmarks2:
    x = landmarks2[lm][:,0]
    y = landmarks2[lm][:,1]
    ax3.plot(x, label=f'x_{lm}')
    ax3.plot(y, label=f'y_{lm}')
    ax3.legend()
plt.show()