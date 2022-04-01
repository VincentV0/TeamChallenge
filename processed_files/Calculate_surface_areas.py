from scipy.spatial import ConvexHull
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.morphology import disk, erosion, dilation, area_closing, square, remove_small_objects, h_maxima
from skimage.measure import regionprops
from skimage import measure
#import utils_surface
from scipy.spatial import ConvexHull, convex_hull_plot_2d


##Functions from LM1368_utils.py
def set_is_closed(contour):
    if contour_distance(contour) < 1:
        return True
    else:
        return False

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

def intensity_seg(ct_numpy, min, max):
    clipped = ct_numpy.clip(min, max)
    clipped[clipped != max] = 1
    clipped[clipped == max] = 0
    return measure.find_contours(clipped, 0.95)

#Adapted
def find_lungs_adapted(contours):
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

### Specific functions
def isolate_sternum(image):
    "Isolate the sternum from an image"
    #Thresholding
    thresh = 210
    th_image = image > thresh
    plt.imshow(th_image)
    plt.show()
    
    
    dl_image = area_closing(th_image)
    #Dilate to connect all bones of the sternum
    dl_image = dilation(dl_image,square(3))
    #Remove ribs from image
    dl_image = remove_small_objects(dl_image, min_size=1500)
    #Undo the dilation
    dl_image = erosion(dl_image,square(3))
    return dl_image

def sternum_surface(image):
    dl_image = dilation(image,disk(15))
    er_image = erosion(dl_image,disk(15))
    
    #plt.imshow(er_image)
    #plt.show()
        
    pixels = np.count_nonzero(er_image)
    return pixels



#Load the image
path = 'C://Users//nibob//OneDrive//Documenten//Tue//Master//TeamCH_P2//Scoliose//Scoliose//'
file = '1postop.nii'
img = nib.load(path+file)
img_data = img.get_data()

sx, sy, sz = img.header.get_zooms()
pixel_area = sx * sy

#Slice selection
image = img_data[:,:,200]



#Regular image
plt.imshow(image)
plt.show()

st_image = isolate_sternum(image)
plt.imshow(st_image)
plt.show()

surface2 = sternum_surface(st_image)
print("Number of pixels in surface 2 ", surface2)
print("Area of surface 2 ", surface2*pixel_area, "mm^2")


contours = intensity_seg(image, min=-1000, max=-300) # Hounsfields units
contours = find_lungs_adapted(contours)
contours_total = np.concatenate((contours[0], contours[1]))
hull = ConvexHull(contours_total)

surface12 = round(hull.volume)
print("Number of pixels for surface 1 and 2 combined ", surface12)

surface1 = surface12 - surface2
print("Number of pixels in surface 1 ", surface1)
print("Area of surface 1 ", surface1*pixel_area, "mm^2")



convex_hull_plot_2d(hull)
plt.imshow(image)
plt.show()
