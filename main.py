"""
Obtaining thoracic landmarks and parameters from CT image.

Project of Team Challenge 2022 - Group 4.

Last updated: 21 Mar 2022
"""

# Import general libraries
import os
import numpy as np
import matplotlib.pyplot as plt

# Import specific code
from readData      import *
from LM12368_utils import load_LM12368
from LM57_utils    import load_LM57
from utils         import rotate, rotate_landmarks, sagittal_diameter, haller_index
from ScrollPlot    import ScrollPlot
from interp_outliers import find_outliers, interpol_alt
from surface_areas import get_surfaces

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
img,header      = read_nii(nifty_path, rotate=True)
img_no_rotate,_ = read_nii(nifty_path, rotate=False)
true_markers    = read_xml(xml_path, rotate=True)
mask1, mask2    = read_nii_masks(mask1_path, mask2_path, rotate=True)

# Setting some variables:
image_origin = (img.shape[0]//2, img.shape[1]//2)
landmarks = dict()
#...

# Rotate the true markers 
true_markers = rotate_landmarks(true_markers, image_origin)



### PART 1 - Landmarks 1, 2, 3, 6 and 8; lung contours
print('\nStarting part 1: Loading landmarks 1, 2, 3, 6, 8 and lung contours')
landmarks[1], landmarks[2], landmarks[3], landmarks[6], landmarks[8], \
    lung_segmentation = load_LM12368(img)
print('Part 1 finished.')



### PART 2 - Landmark 5
print('\nStarting part 2: Loading landmark 5')
landmarks[5], _, dl_image = load_LM57(img_no_rotate, postop)

# Rotate point 5 to new frame of reference.
landmarks[5] = rotate_landmarks(landmarks[5], image_origin)
print('Part 2 finished.')



### PART 3 - Filter outliers and perform interpolation on unknown points.
print('\nStarting part 3: Filtering and interpolating outliers')
for lm in landmarks:
    x_list = landmarks[lm][:,0]
    y_list = landmarks[lm][:,1]
    reference = true_markers[lm-1][2]
    threshold = 200
    landmarks[lm][:,0], landmarks[lm][:,1] = find_outliers(x_list, y_list, reference, threshold)
print('Part 3 finished.')



### PART 4 - Landmarks 4 and 7 (manually)
# Pick landmark 4 on a couple of slices
print('\nPart 4: Selection of landmarks 4 and 7')
fig4, ax4 = plt.subplots()
sp4 = ScrollPlot(ax4, img, None, landmarks, true_markers, ax_title="Select landmark 4 on a reasonable number of slices")
fig4.canvas.mpl_connect('scroll_event', sp4.on_scroll)
fig4.canvas.mpl_connect('button_press_event', sp4.on_click)
plt.show()
points_selected_LM4 = sp4.get_marked_points()

# Pick landmark 7 on a couple of slices and interpolate
fig7, ax7 = plt.subplots()
sp7 = ScrollPlot(ax7, img, None, landmarks, true_markers, ax_title="Select landmark 7 on a reasonable number of slices")
fig7.canvas.mpl_connect('scroll_event', sp7.on_scroll)
fig7.canvas.mpl_connect('button_press_event', sp7.on_click)
plt.show()
points_selected_LM7 = sp7.get_marked_points()

# Convert point_selected_LM4/-7 to different format
x4,y4 = np.ones((img.shape[2]))*-1, np.ones((img.shape[2]))*-1
x7,y7 = np.ones((img.shape[2]))*-1, np.ones((img.shape[2]))*-1

for point in points_selected_LM4:
    x4[point[2]] = point[0]
    y4[point[2]] = point[1]
for point in points_selected_LM7:
    x7[point[2]] = point[0]
    y7[point[2]] = point[1]

x4i,y4i = interpol_alt(x4,y4)
x7i,y7i = interpol_alt(x7,y7)

landmarks[4] = np.transpose(np.stack((x4i,y4i)))
landmarks[7] = np.transpose(np.stack((x7i,y7i)))
print('Part 4 finished.')


### PART 5 - Areas (unit = mm^2)
print('\nStarting part 5: Loading surface areas')
surf1, surf2 = get_surfaces(img, header)
print('Part 5 finished.')


### PART 6 - Thoracic parameters
print('\nStarting part 6: Calculating thoracic parameters')
HI = haller_index(landmarks)
print('Part 6 finished.')

# Make ScrollPlot
fig1, ax1 = plt.subplots()
sp1 = ScrollPlot(ax1, img, None, landmarks, true_markers)
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

