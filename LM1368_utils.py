"""
Code is provided by Adaloglou Nikolas (2021)

https://github.com/black0017/ct-intensity-segmentation

"""
import os
import cv2
import math
# import shutil

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from skimage.segmentation import flood, flood_fill
from skimage import measure, filters, color, morphology


def show_slice(slice):
    """
    Function to display an image slice
    Input is a numpy 2D array
    """
    plt.figure()
    plt.imshow(slice.T, cmap="gray", origin="lower")

def overlay_plot(im, mask):
    plt.figure()
    plt.imshow(im.T, 'gray', interpolation='none')
    plt.imshow(mask.T, 'jet', interpolation='none', alpha=0.5)
    
# def make_dirs(path):
#     """
#     Creates the directory as specified from the path
#     in case it exists it deletes it
#     """
#     if os.path.exists(path):
#         shutil.rmtree(path)
#         os.mkdir(path)
#     else:
#         os.makedirs(path)

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
    """
    Chooses the contours that correspond to the lungs and the body
    FIrst we exclude non closed sets-contours
    Then we assume some min area and volume to exclude small contours
    Then the body is excluded as the highest volume closed set
    The remaining areas correspond to the lungs

    Args:
        contours: all the detected contours

    Returns: contours that correspond to the lung area

    """
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
    """
    Creates a binary mask with the dimensions of the image and
    converts the list of polygon-contours to binary masks and merges them together
    Args:
        image: the image that the contours refer to
        contours: list of contours

    Returns:

    """
    lung_mask = np.array(Image.new('L', image.shape, 0))
    for contour in contours:
        x = contour[:, 0]
        y = contour[:, 1]
        polygon_tuple = list(zip(x, y))
        img = Image.new('L', image.shape, 0)
        ImageDraw.Draw(img).polygon(polygon_tuple, outline=0, fill=1)
        mask = np.array(img)
        lung_mask += mask

    lung_mask[lung_mask > 1] = 1  # sanity check to make 100% sure that the mask is binary

    return lung_mask.T  # transpose it to be aligned with the image dims

# def save_nifty(img_np, name, affine):
#     """
#     binary masks should be converted to 255 so it can be displayed in a nii viewer
#     we pass the affine of the initial image to make sure it exits in the same
#     image coordinate space
#     Args:
#         img_np: the binary mask
#         name: output name
#         affine: 4x4 np array
#     Returns:
#     """
#     img_np[img_np == 1] = 255
#     ni_img = nib.Nifti1Image(img_np, affine)
#     nib.save(ni_img, name + '.nii.gz')

# def find_pix_dim(ct_img):
#     """
#     Get the pixdim of the CT image.
#     A general solution that get the pixdim indicated from the image
#     dimensions. From the last 2 image dimensions we get their pixel dimension.
#     Args:
#         ct_img: nib image

#     Returns: List of the 2 pixel dimensions
#     """
#     pix_dim = ct_img.header["pixdim"]
#     dim = ct_img.header["dim"]
#     max_indx = np.argmax(dim)
#     pixdimX = pix_dim[max_indx]
#     dim = np.delete(dim, max_indx)
#     pix_dim = np.delete(pix_dim, max_indx)
#     max_indy = np.argmax(dim)
#     pixdimY = pix_dim[max_indy]
#     return [pixdimX, pixdimY]

def contour_distance(contour):
    """
    Given a set of points that may describe a contour
     it calculates the distance between the first and the last point
     to infer if the set is closed.
    Args:
        contour: np array of x and y points

    Returns: euclidean distance of first and last point
    """
    dx = contour[0, 1] - contour[-1, 1]
    dy = contour[0, 0] - contour[-1, 0]
    return np.sqrt(np.power(dx, 2) + np.power(dy, 2))

def compute_area(mask, pixdim):
    """
    Computes the area (number of pixels) of a binary mask and multiplies the pixels
    with the pixel dimension of the acquired CT image
    Args:
        lung_mask: binary lung mask
        pixdim: list or tuple with two values

    Returns: the lung area in mm^2
    """
    mask[mask >= 1] = 1
    lung_pixels = np.sum(mask)
    return lung_pixels * pixdim[0] * pixdim[1]

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
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


def load_LM1368(ct_img, plot=False):
    """
    Given an CT image, the landmark positions are calculated 
     and a segmentation of the lungs is made.
    Args:
        ct_img: np array of the CT image (90 degrees rotated)

    Returns:

    """
    lung_seg = np.zeros(ct_img.shape)
    empty_slices = np.array([])
    surface1 = np.zeros(ct_img.shape)

    p1 = np.ones((ct_img.shape[2],2)) * -1    # when no lung is found, no landmarks can be found (= -1)
    p3 = np.ones((ct_img.shape[2],2)) * -1
    p6 = np.ones((ct_img.shape[2],2)) * -1
    p8 = np.ones((ct_img.shape[2],2)) * -1

    for slice in tqdm(range(ct_img.shape[2])):
        ct_numpy = ct_img[:,:,slice]

        # Get the coordinates of the lung countours and create the mask
        contours = intensity_seg(ct_numpy, min=-1000, max=-300) # Hounsfields units
        lung_contours = find_lungs(contours)
        
        if not lung_contours:
            empty_slices = np.append(empty_slices,slice)
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

        # Connect the lungs for floodfill
        line_1 = get_line(x_LL[idx_top_LL],y_LL[idx_top_LL],x_RL[idx_top_RL],y_RL[idx_top_RL])
        line_2 = get_line(p6[slice,0],p6[slice,1],p8[slice,0],p8[slice,1])

        for y,x in line_1:
            for i in range(10):
                lung_mask[y+i,x] = 1

        for y,x in line_2:
            for i in range(10):
                lung_mask[y-i,x] = 1
                
        # Get center of the surface
        center_x = round((p1[slice,0]+p3[slice,0])/2)
        center_y = int(p3[slice,1])

        # Apply morphological closing
        kernel_size = (10,10)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,kernel_size)
        area = flood_fill(lung_mask, (center_x,center_y), 1)
        surface1[:,:,slice] = cv2.morphologyEx(area, cv2.MORPH_CLOSE, kernel)

        if plot:
            # ax[1].imshow(lung_mask, cmap='gray')
            ax[1].imshow(surface1, cmap='gray')
            ax[1].set_title('Surface area 1')
            plt.show()

    return p1,p3,p6,p8,surface1,lung_seg,empty_slices
