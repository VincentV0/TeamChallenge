import os

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from readData      import *
from LM1368_utils  import load_LM1368
from LM57_utils    import load_LM57
from ScrollPlot    import ScrollPlot

DATA_PATH = r"C:\\temp\\TC_DATA\\Scoliose"

print("Please enter the image ID that should be processed: ")
image_ID = input('> ')
if image_ID[1:] == 'postop': postop = True
else: postop = False


nifty_path = os.path.join(DATA_PATH, image_ID + ".nii")
xml_path   = os.path.join(DATA_PATH, image_ID + ".xml")
mask1_path = os.path.join(DATA_PATH, image_ID + "_mask1.nii")
mask2_path = os.path.join(DATA_PATH, image_ID + "_mask2.nii")

print('\nLoading images and manual annotations (for validation)...')
img            = read_nii(nifty_path, rotate=True)
img_no_rotate  = read_nii(nifty_path, rotate=False)
manual_markers = read_xml(xml_path, rotate=True)
mask1, mask2   = read_nii_masks(mask1_path, mask2_path, rotate=True)


# Setting some variables:
landmarks = dict()
#...

print('\nStarting part 1: Loading landmarks 1, 3, 6 and 8, lung contours and surface 1...')
landmarks[1], landmarks[3], landmarks[6], landmarks[8], \
    surface1, lung_segmentation, empty_slices = load_LM1368(img)
print('Part 1 finished.')

print('\nStarting part 2: Loading landmarks 5 and 7')
landmarks[5], landmarks[7], dl_image = load_LM57(img_no_rotate, postop)
print('Part 2 finished.')

# Make ScrollPlot
fig1, ax1 = plt.subplots()
sp1 = ScrollPlot(ax1, img, None, landmarks)
fig1.canvas.mpl_connect('scroll_event', sp1.on_scroll)
fig1.canvas.mpl_connect('button_press_event', sp1.on_click)
plt.show()
point_coords = sp1.get_marked_points()

if point_coords.size > 0:
    print("Selected points:")
    print("{:10} {:10} {:10}".format('x','y','z'))
    for x,y,z in point_coords:
        print("{:10} {:10} {:10}".format(x,y,z))