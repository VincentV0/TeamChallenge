from skimage.morphology import erosion, dilation, area_closing, square, remove_small_objects
import numpy as np
from tqdm import tqdm


# Set parameters
MIN_THRESHOLD_BONE = 210
MAX_THRESHOLD_ANTI_METAL = 800 # for post-op images only
MIN_THRESHOLD_STRUCTURE_SIZE = 1500

def LM5(image):
    """
    Determines the coordinates for point 5 by analyzing the binary
    mask of the spine
    """
    y=0
    x=0
    while y < image.shape[1]-1:
        for x in range(image.shape[0] - 1 ):
            if image[x,y] == True:
                point_5 = [y,x]
                break
            x=x+1
        y=y+1
    return point_5


def LM7(dl_image):
    """
    Determines the coordinates for point 7 by analyzing the binary
    mask of the spine
    """
    spine=np.nonzero(dl_image) #indicate nonzeros in binary image with spine as nonzeros

    spine_x=spine[1] #select x coordinates of spine
    spine_y=spine[0] #select y coordinates of spine
    canal_x=[] #empty array to store x coordinates of spine
    canal_y=[] # empty array to store y coordinates of spine

    for i in range (len(spine_x)-1):
        if spine_y[i]==spine_y[i+1]: # check if preceding y coordinates are the same
            difference=spine_x[i+1]-spine_x[i] # calculate difference in preceding x coordinates
            if abs(difference) >20: # If difference is larger than 20, coordinates are stored
                canal_x.append(spine_x[i])
                canal_y.append(spine_y[i])
    canal_x=np.array(canal_x)# make array of x coordinates
    canal_y=np.array(canal_y) # make array of y coordinates
    if canal_x.size == 0 and canal_y.size == 0:
        return -1, -1 # no canal found
    else:
        min_value_x=np.min(canal_x) # minimal vallue for x to select posterior point
        min_index=np.where(canal_x==min_value_x) #index of minimal value of x

        min_value_y=canal_y[min_index] # corresponding y coordinates of minimal value of x
        min_value_y=min(min_value_y) # minimal value of y is the desired point
        return min_value_x, min_value_y


def load_LM57(img_data, postop=False):
    # Screw reduction (for post-op images only)
    if postop:
        img_data[img_data > MAX_THRESHOLD_ANTI_METAL] = MAX_THRESHOLD_ANTI_METAL

    # Create empty arrays to store data in
    dl_image_3d = np.zeros(img_data.shape)
    p5 = np.ones((img_data.shape[2],2)) * -1
    p7 = np.ones((img_data.shape[2],2)) * -1

    for slice in tqdm(range(img_data.shape[2])):
        # Slice selection
        image = img_data[:,:,slice]

        # Thresholding
        th_image = image > MIN_THRESHOLD_BONE

        # Sternum isolation
        dl_image = area_closing(th_image)
        # Dilate to connect all bones of the sternum
        dl_image = dilation(dl_image,square(3))
        # Remove ribs from image
        dl_image = remove_small_objects(dl_image, min_size=1500)
        # Undo the dilation
        dl_image = erosion(dl_image,square(3))


        p5[slice] = LM5(dl_image)
        p7[slice] = LM7(dl_image)

        # Store 3D data
        dl_image_3d[:,:,slice] = dl_image
    return p5, p7, dl_image_3d
