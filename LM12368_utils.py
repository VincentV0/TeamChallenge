"""
Code is provided by Adaloglou Nikolas (2021)

https://github.com/black0017/ct-intensity-segmentation

"""
import cv2
import math

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from skimage import measure


HU_BONE = 200 # bone is around 300-400 HU (taking artefacts such as streaking- and cupping artefacts in account so decreasing to 200)

def thresholded_bone(ct_image_2d):
    
    _,thresh = cv2.threshold(ct_image_2d, HU_BONE, 3000, cv2.THRESH_BINARY)
    thres = thresh.astype('int16')
    
    bone_threshold_filtered_image = cv2.medianBlur(thres, 3)
    return bone_threshold_filtered_image

def intensity_seg(ct_numpy, min, max):
    
    clipped = ct_numpy.clip(min, max)
    clipped[clipped != max] = 1
    clipped[clipped == max] = 0
    return measure.find_contours(clipped, 0.95)

def set_is_closed(contour):
    
    if contour_distance(contour) < 1:
        return True
    else:
        return False

def find_lungs(contours):
    
    body_and_lung_contours = []
    vol_contours = []

    for contour in contours:
        hull = ConvexHull(contour)

        if hull.volume > 2000 and set_is_closed(contour):
            body_and_lung_contours.append(contour)
            vol_contours.append(hull.volume)

    if len(body_and_lung_contours) == 2:
        return body_and_lung_contours
    elif len(body_and_lung_contours) > 2:
        vol_contours, body_and_lung_contours = (list(t) for t in
                                                zip(*sorted(zip(vol_contours, body_and_lung_contours))))
        body_and_lung_contours.pop(-1)
        return body_and_lung_contours


def show_contour(image, contours, name=None, save=False):
    
    fig, ax = plt.subplots()
    ax.imshow(image.T, cmap=plt.cm.gray)
    for contour in contours:
        ax.plot(contour[:, 0], contour[:, 1], linewidth=1)

    ax.set_xticks([])
    ax.set_yticks([])

    if save:
        plt.savefig(name)
        plt.close(fig)
    else:
        plt.show()

def create_mask_from_polygon(image, contours):
    
    lung_mask = np.array(Image.new('L', image.shape, 0))
    for contour in contours:
        x = contour[:, 0]
        y = contour[:, 1]
        polygon_tuple = list(zip(x, y))
        img = Image.new('L', image.shape, 0)
        ImageDraw.Draw(img).polygon(polygon_tuple, outline=0, fill=1)
        mask = np.array(img)
        lung_mask += mask

    lung_mask[lung_mask > 1] = 1  

    return lung_mask.T  

def contour_distance(contour):
    
    dx = contour[0, 1] - contour[-1, 1]
    dy = contour[0, 0] - contour[-1, 0]
    
    return np.sqrt(np.power(dx, 2) + np.power(dy, 2))

def compute_area(mask, pixdim):

    mask[mask >= 1] = 1
    lung_pixels = np.sum(mask)
    
    return lung_pixels * pixdim[0] * pixdim[1]

def rotate(origin, point, angle):

    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    
    return qx, qy

def get_line(x1,y1,x2,y2):

    line = []
    
    a = (y1-y2)/(x1-x2) # Slope
    b = (x1*y2 - x2*y1)/(x1-x2)
    for x in range(int(x1),int(x2)):
        y = a*x + b
        line.append([round(y),x])
         
    return line

def centroid_sternum(bone_threshold_filtered_image, l1, l2):
    
    # Apply mask within the field of view l1 and l2
    mask = np.zeros(bone_threshold_filtered_image.shape)
    mask[l1[1]-50:l1[1]+50,l1[0]:l2[0]] = 1
    
    middle_col = np.around((l1[0]+l2[0])/2)
    fov_im90 = bone_threshold_filtered_image*mask

    # Apply morphological closing
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(fov_im90, cv2.MORPH_CLOSE, kernel, iterations=3)
    closing = closing.astype('int8')# om te voorkomen dat je een of andere vage error over orde van grootte van de getallen krijgt

    # Extract sternum with properties
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

    # find indices where the value changes
    index_backgroud_mask = np.where(centroid_axis_corr[:-1] != centroid_axis_corr[1:])[0] 
    
    # add the value of the first index to the centroid row value
    posterior_centroid_sternum_row = index_backgroud_mask[0]+row_centroid_sternum
    
    return col_centroid_sternum, posterior_centroid_sternum_row


def load_LM12368(ct_img, plot=False):
    """
    Given an CT image, the landmark positions are calculated 
     and a segmentation of the lungs is made.
    Args:
        ct_img: np array of the CT image

    Returns:
        landmarks 1, 2, 3, 6, 8, lung_segmentation

    """
    lung_seg = np.zeros(ct_img.shape)

    p1 = np.ones((ct_img.shape[2],2)) * -1    # when no lung is found, no landmarks can be found (= -1)
    p2 = np.ones((ct_img.shape[2],2)) * -1    
    p3 = np.ones((ct_img.shape[2],2)) * -1
    p6 = np.ones((ct_img.shape[2],2)) * -1
    p8 = np.ones((ct_img.shape[2],2)) * -1

    for slice in tqdm(range(ct_img.shape[2])):
        ct_numpy = ct_img[:,:,slice]

        # Get the coordinates of the lung countours and create the mask
        contours = intensity_seg(ct_numpy, min=-1000, max=-300) # Hounsfields units
        lung_contours = find_lungs(contours)
        
        if not lung_contours:
            continue # skip the rest of the steps and continue with the next slice
        
        lung_mask = np.array(create_mask_from_polygon(ct_numpy, lung_contours))
        lung_seg[:,:,slice] = lung_mask

        # Separate the left and right lung
        left_lung, right_lung = np.rot90(lung_contours[0],2).astype(int), np.rot90(lung_contours[1],2).astype(int)

        # Check if the labeling of the lung is correct
        if left_lung[0,0] > right_lung[0,0]: # first coordinate is the most bottom maximum in the lungs
            left_lung, right_lung = right_lung, left_lung  # switch variables
            
        # Get maxima
        x_LL, y_LL = left_lung[:, 0], left_lung[:, 1]
        x_RL, y_RL = right_lung[:, 0], right_lung[:, 1]

        idx_bottom_LL = np.argmax(y_LL)
        idx_top_LL = np.argmin(y_LL)
        idx_left_LL = np.argmin(x_LL)
        idx_right_LL = np.argmax(x_LL)

        idx_bottom_RL = np.argmax(y_RL)
        idx_top_RL = np.argmin(y_RL)
        idx_left_RL = np.argmin(x_RL)
        idx_right_RL = np.argmax(x_RL)

        # Landmarks
        p3[slice] = x_RL[idx_right_RL], y_RL[idx_right_RL]
        p1[slice] = x_LL[idx_left_LL], y_LL[idx_left_LL]
        p6[slice] = x_LL[idx_bottom_LL], y_LL[idx_bottom_LL]
        p8[slice] = x_RL[idx_bottom_RL], y_RL[idx_bottom_RL]
        
        # Most anterior lung coordinates
        l1 = x_LL[idx_top_LL], y_LL[idx_top_LL]
        l2 = x_RL[idx_top_RL], y_RL[idx_top_RL]
        
        # Store the posterior midpoint of the sternum
        try:
            if np.sum(lung_mask) > 0:
                bone_threshold_filtered_image = thresholded_bone(ct_numpy)
                p2[slice] = centroid_sternum(bone_threshold_filtered_image, l1, l2)
        except:
            pass
        
        if plot:
            # Plot the segmented lungs with landmarks
            fig, ax = plt.subplots(1, 2, figsize=(20, 5))
            ax[0].imshow(lung_mask, cmap='gray')
            ax[0].set_title('Segmented lungs with landmarks')
            ax[0].plot(p1[slice,0], p1[slice,1], marker='o', markersize=3, color="red")
            ax[0].plot(p3[slice,0], p3[slice,1], marker='o', markersize=3, color="yellow")
            ax[0].plot(p6[slice,0], p6[slice,1], marker='o', markersize=3, color="blue")
            ax[0].plot(p8[slice,0], p8[slice,1], marker='o', markersize=3, color="green")

            # # Other maxima
            # ax[0].plot(x_LL[idx_top_LL], y_LL[idx_top_LL], marker='o', markersize=3, color="lightblue")
            # ax[0].plot(x_LL[idx_right_LL], y_LL[idx_right_LL], marker='o', markersize=3, color="purple")
            # ax[0].plot(x_RL[idx_top_RL], y_RL[idx_top_RL], marker='o', markersize=3, color="lightgreen")
            # ax[0].plot(x_RL[idx_left_RL], y_RL[idx_left_RL], marker='o', markersize=3, color="orange")


    return p1,p2,p3,p6,p8,lung_seg
