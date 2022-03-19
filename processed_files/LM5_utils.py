import nibabel as nib
import matplotlib.pyplot as plt
from skimage.morphology import disk, erosion, dilation, area_closing, square, remove_small_objects, h_maxima
from skimage.measure import regionprops
import numpy as np
from tqdm import tqdm


THRESHOLD = 210

def find_sternum(img, center_of_mass):
    nonzero = np.argwhere(img == True)
    distances = np.sqrt((nonzero[:,0] - center_of_mass[0]) ** 2 + (nonzero[:,1] - center_of_mass[1]) ** 2)
    nearest_index = np.argmin(distances)
    return nonzero[nearest_index]


def find_point5(image):
#Find location of point 5
#By using the processed image with only the sternum visible
    y=0
    x=0
    while y < image.shape[1]-1:
        for x in range(image.shape[0] - 1 ):
            if image[x,y] == True:
                point_5 = [x,y]
                break
            x=x+1
        y=y+1
    return point_5

def load_LM5(image_3d):
    p5 = np.ones((image_3d.shape[2],2)) * -1    # when no point found, -1
    
    # Rotate image back
    image_3d_rot = np.rot90(image_3d, k=1, axes=(1,0))

    for slice in tqdm(range(image_3d_rot.shape[2])):
        image = image_3d[:,:,slice]

        th_image = image > THRESHOLD

        # Sternum isolation
        dl_image = area_closing(th_image)
        # Dilate to connect all bones of the sternum
        dl_image = dilation(dl_image,square(3))
        # Remove ribs from image
        dl_image = remove_small_objects(dl_image, min_size=1500)
        # Undo the dilation
        dl_image = erosion(dl_image,square(3))

        ## Select center of mass for an image with just the sternum,
        ## to use as reference for images with multiple structures

        labeled_foreground = dl_image.astype(int)
        properties = regionprops(labeled_foreground)
        center_of_mass = properties[0].centroid

        ################ Fill the image to get rid of holes
        dl_image = dilation(dl_image,disk(15))
        dl_image = erosion(dl_image,disk(15))

        point_5 = find_point5(dl_image)
        p5[slice] = point_5[1], point_5[0] # rotate backx
    return p5
