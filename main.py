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
from readData         import *
from LM12368_utils    import load_LM12368
from LM57_utils       import load_LM57
from utils            import haller_index_export, export_to_excel, rotate, rotate_landmarks, sagittal_diameter, haller_index
from ScrollPlot       import ScrollPlot
from interp_outliers  import find_outliers, interpol_alt
from surface_areas    import get_surfaces
from manual_landmarks import manual_translation

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

print("Would you like to compare two images, or process a single one?")
option1 = input("[1/2] > ")

if option1 == '1':
    # Ask the user for the image ID. As described above, this should be something like "1postop"
    print("Please enter the image ID that should be processed: ")
    image_ID = input('> ')
    if image_ID[1:] == 'postop': postop = True
    else: postop = False
    all_postop = [postop]

elif option1 == '2':
    # Ask the user for the image ID. As described above, this should be something like "1postop"
    print("Please enter the image IDs that should be processed: ")
    image_ID1 = input('> ')
    image_ID2 = input('> ')
    if image_ID1[1:] == 'postop': postop1 = True
    else: postop1 = False
    if image_ID2[1:] == 'postop': postop2 = True
    else: postop2 = False
    all_postop = [postop1, postop2]
else:
    print("Unknown option. Exiting")
    exit()

# Ask if surfaces should be determined
surfaces_question = 'None'
while surfaces_question.lower() != '' and surfaces_question.lower() != 'y' and surfaces_question.lower() != 'n':
    print("Do you want to calculate the surfaces? Be aware that this may take more than 1 hour!")
    surfaces_question = input('y/[n] > ')
    if surfaces_question.lower() == 'y':
        load_surfaces=True
    elif surfaces_question != 'n' and surfaces_question != '':
        print("Unknown option, try again.")
    else:
        load_surfaces=False

# Load data
print('\nLoading images and manual annotations (for validation)...')
if option1 == '1':
    # Load the .nii and .xml files
    nifty_path = os.path.join(DATA_PATH, image_ID + ".nii")
    xml_path   = os.path.join(DATA_PATH, image_ID + ".xml")

    img,header           = read_nii(nifty_path, rotate=True)
    img_no_rotate,_      = read_nii(nifty_path, rotate=False)
    image_origin         = (img.shape[0]//2, img.shape[1]//2)

    true_markers         = read_xml(xml_path, rotate=True)
    true_markers         = rotate_landmarks(true_markers, image_origin)

    # Combine data for loop
    all_imgs = [img]
    all_headers = [header]
    all_tm = [true_markers]
    all_imgs_no_rotate = [img_no_rotate]
    all_origins = [image_origin]

    
if option1 == '2':
    # Load the .nii and .xml files
    nifty_path1 = os.path.join(DATA_PATH, image_ID1 + ".nii")
    nifty_path2 = os.path.join(DATA_PATH, image_ID2 + ".nii")
    xml_path1   = os.path.join(DATA_PATH, image_ID1 + ".xml")
    xml_path2   = os.path.join(DATA_PATH, image_ID2 + ".xml")

    img1_no_rotate,header1 = read_nii(nifty_path1, rotate=False)
    img2_no_rotate,header2 = read_nii(nifty_path2, rotate=False)
    image1_origin = (img1_no_rotate.shape[0]//2, img1_no_rotate.shape[1]//2)
    image2_origin = (img2_no_rotate.shape[0]//2, img2_no_rotate.shape[1]//2)

    true_markers1    = read_xml(xml_path1, rotate=True)
    true_markers2    = read_xml(xml_path2, rotate=True)

    # Do translation of second image and do rotation    
    img2_no_rotate, true_markers2 = manual_translation(img1_no_rotate, img2_no_rotate, true_markers2)
    true_markers1 = rotate_landmarks(true_markers1, image1_origin)
    true_markers2 = rotate_landmarks(true_markers2, image2_origin)

    img1 = np.rot90(img1_no_rotate)
    img2 = np.rot90(img2_no_rotate)

    # Combine data for loop
    all_imgs = [img1, img2]
    all_headers = [header1, header2]
    all_tm = [true_markers1, true_markers2]
    all_imgs_no_rotate = [img1_no_rotate, img2_no_rotate]
    all_origins = [image1_origin, image2_origin]


for phase, (img, header, true_markers, img_no_rotate, postop, image_origin) \
    in enumerate(zip(all_imgs, all_headers, all_tm, all_imgs_no_rotate, all_postop, all_origins)):

    # Setting the landmark variable:
    landmarks = dict()
    

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

    # Pick landmark 7 on a couple of slices
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

    # Interpolate selected points
    x4i,y4i = interpol_alt(x4,y4)
    x7i,y7i = interpol_alt(x7,y7)

    landmarks[4] = np.transpose(np.stack((x4i,y4i)))
    landmarks[7] = np.transpose(np.stack((x7i,y7i)))
    print('Part 4 finished.')


    ### PART 5 - Areas (unit = mm^2)
    if load_surfaces:
        print('\nStarting part 5: Loading surface areas')
        surfaces1, surfaces2, surface1_mm2, surface2_mm2 = get_surfaces(img, header)
        print('Part 5 finished.')


    ### PART 6 - Thoracic parameters
    print('\nStarting part 6: Calculating thoracic parameters')
    HI = haller_index(landmarks, header)
    print('Part 6 finished.')

    ### PART 7 - Exporting landmarks
    export_q = 'None'
    while export_q.lower() != '' and export_q.lower() != 'y' and export_q.lower() != 'n':
        print("\nDo you want to export the landmarks to an .xlsx file?")
        export_q = input('y/[n] > ')
        if export_q.lower() == 'y':
            print("Enter filename for xlsx-file: ")
            xlsx_path = input("> ")
            if xlsx_path[-5:] != '.xlsx': 
                xlsx_path += '.xlsx' # add extension if necessary
            export_to_excel(landmarks, xlsx_path)
        elif export_q != 'n' and export_q != '':
            print("Unknown option, try again.")
        else:
            pass

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

    # Save the Haller index arrays
    if phase==0:
        landmarks1 = landmarks.copy()
        HI1 = HI.copy()
    if phase==1:
        landmarks2 = landmarks.copy()
        HI2 = HI.copy()

# Export Haller indices to xlsx-file
export_HI_q = 'None'
while export_HI_q.lower() != '' and export_HI_q.lower() != 'y' and export_HI_q.lower() != 'n':
    print("\nDo you want to export the Haller indices to an .xlsx file?")
    export_HI_q = input('y/[n] > ')
    if export_HI_q.lower() == 'y':
        print("Enter filename for xlsx-file: ")
        xlsx_path = input("> ")
        if xlsx_path[-5:] != '.xlsx': 
            xlsx_path += '.xlsx' # add extension if necessary
        if option1=='2':
            haller_index_export(HI1, HI2, filename=xlsx_path)
        elif option1=='1':
            haller_index_export(HI1, filename=xlsx_path)
    elif export_HI_q != 'n' and export_HI_q != '':
        print("Unknown option, try again.")
    else:
        pass

