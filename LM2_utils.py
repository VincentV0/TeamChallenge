import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

HU_BONE = 200 # bone is around 300-400 HU (taking artefacts such as streaking- and cupping artefacts in account so decreasing to 200)

def thresholded_bone(ct_image_2d):
    _,thresh = cv2.threshold(ct_image_2d, HU_BONE, 3000, cv2.THRESH_BINARY)
    thres = thresh.astype('int16') # om te voorkomen dat je een of andere vage error over orde van grootte van de getallen krijgt
    
    bone_threshold_filtered_image = cv2.medianBlur(thres, 3)
    return bone_threshold_filtered_image

def find_most_anterior_lung_points(lung_mask):
    # Check if the image is correctly oriented e.g. the sternum at the top, vertebrea at the bottom of the image,
    lung_mask = np.uint8(lung_mask)
    arr = np.nonzero(lung_mask)
    l1 = [arr[0][0]],[arr[1][0]]
    
    # Marker labelling to to able to distinct between masks
    ret, markers = cv2.connectedComponents(lung_mask)

    # één masker verwijderen
    markers[markers == markers[l1]] = 0 # markers in markers die ongelijk zijn aan de waarde van het eerste punt 
    arr2 = np.nonzero(markers)
    l2 = [arr2[0][0]],[arr2[1][0]] 
    return l1, l2

def centroid_sternum(lung_mask_90, bone_threshold_filtered_image, l1, l2):
    # set values of columns before the first point and points after column of the second point to zero (of rotated original image)
    fov_between_lungs = l1[1], l2[1]
    fov_between_lungs = np.sort(fov_between_lungs, axis=None)
    rows = np.prod(lung_mask_90[1].shape)
    bone_threshold_filtered_image_90 = np.rot90(bone_threshold_filtered_image)
    mask = np.zeros(lung_mask_90.shape)
    mask[l1[0][0]-50:l1[0][0]+50,fov_between_lungs[0]:fov_between_lungs[1]]=1

    middle_col = np.around((fov_between_lungs[0]+fov_between_lungs[1])/2)
    # middle_row = np.around()
    #print(fov_between_lungs[0],fov_between_lungs[1])
    fov_im90 = bone_threshold_filtered_image_90*mask
#     imgplot = plt.imshow(fov_im90, cmap='gray')
    # plt.plot(middle_point, 140, marker='o', markersize=5, color="blue") # middle of the two lung points

    # posterior middelpunt sternum bepalen:

    #closing uitvoeren
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(fov_im90, cv2.MORPH_CLOSE, kernel, iterations=3)
    closing = closing.astype('int8')# om te voorkomen dat je een of andere vage error over orde van grootte van de getallen krijgt
    #imgplot = plt.imshow(closing, cmap='gray')


    # Choose 4 or 8 for connectivity type
    connectivity = 4  
    masks = cv2.connectedComponentsWithStats(closing, connectivity, cv2.CV_32S)

    num_labels = masks[0]
    stats = masks[2]
    centroids = masks[3]

    for label in range(1,num_labels):
        blob_area = stats[label, cv2.CC_STAT_AREA]
        blob_width = stats[label, cv2.CC_STAT_WIDTH]
        blob_height = stats[label, cv2.CC_STAT_HEIGHT]

    
    mask_area = stats[:,4]
    filtered_mask_area = np.sort([x for x in mask_area if x <= 100000 and x > 50])
    a = stats[:,4] == filtered_mask_area[0]
    b=[i for i, x in enumerate(a) if x]
    centroid_sternum = np.around(centroids[b])
    
    
    row_centroid_sternum = np.uint(centroid_sternum[0][1])
    col_centroid_sternum = np.uint(centroid_sternum[0][0])
    centroid_axis = closing[:,col_centroid_sternum]
    centroid_axis_corr = centroid_axis[row_centroid_sternum:]

    # find indices where the value changes.
    index_backgroud_mask = np.where(centroid_axis_corr[:-1] != centroid_axis_corr[1:])[0] 
    # add the value of the first index to the centroid row value
    posterior_centroid_sternum_row = index_backgroud_mask[0]+row_centroid_sternum
    
    return posterior_centroid_sternum_row, col_centroid_sternum


def load_LM2(ct_image, lung_mask):
    p2 = np.ones((ct_image.shape[2],2)) * -1    # when no lung is found, no landmarks can be found (= -1)

    for slice in tqdm(range(ct_image.shape[2])):
        try:
            ct_slice = ct_image[:,:,slice]
            lung_mask_2d = lung_mask[:,:,slice]

            if np.sum(lung_mask_2d) > 0:
                l1, l2 = find_most_anterior_lung_points(lung_mask_2d)
                bone_threshold_filtered_image = thresholded_bone(ct_slice)
                p2[slice] = centroid_sternum(lung_mask_2d, bone_threshold_filtered_image, l1, l2)
        except:
            pass
    return p2