import numpy as np

from tqdm import tqdm
from scipy import ndimage
from skimage import measure
from scipy.spatial import ConvexHull
from skimage.measure import regionprops
from skimage.morphology import disk, erosion, dilation, area_closing, square, remove_small_objects

def set_is_closed(contour):
    """
    Check if the contour is closed.
    """
    if contour_distance(contour) < 1:
        return True
    else:
        return False

def contour_distance(contour):
    """
    Given a set of points that may describe a contour
     it calculates the distance between the first and the last point.
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

def isolate_spine(image):
    """"
    Isolate the sternum from an image.
    First the image is thresholded, to retrieve bone structures.
    Then smaller bone structures are removed from the image.
    Followed by using a maximum dinstance from the centroid of the image to
    get the final spine structure.
    
    """
    
    # Thresholding
    thresh = 210
    th_image = image > thresh    
    
    # Morpholocial closing
    dl_image = area_closing(th_image)
    
    # Dilate to connect all bones of the sternum
    dl_image = dilation(dl_image,square(3))
    
    # Remove ribs from image
    dl_image = remove_small_objects(dl_image, min_size=1500)
    
    # Undo the dilation
    dl_image = erosion(dl_image,square(3))
    
    labeled_foreground = dl_image.astype(int)
    properties = regionprops(labeled_foreground)
    center_of_mass = properties[0].centroid
    
    nonzero = np.argwhere(dl_image == True)
    distances = np.sqrt((nonzero[:,0] - center_of_mass[0]) ** 2 + (nonzero[:,1] - center_of_mass[1]) ** 2)
    
    avg_distance = 100 # Chosen based on experimental results
    
    for point in properties[0].coords:
        dist = np.sqrt((point[0] - center_of_mass[0]) ** 2 + (point[1] - center_of_mass[1]) ** 2)
        if dist > avg_distance:
            dl_image[point[0],point[1]]=False
            
    return dl_image, center_of_mass

def spine_surface(image):
    """
    Complete the surface of the spine by getting rid of holes in the structure.
    By means of applying a large area of dilation followed by erosion,
    and a fill holes function for any larger gaps.
    """
    dl_image = dilation(image,disk(15))
    er_image = erosion(dl_image,disk(15))
        
    filled_image = ndimage.binary_fill_holes(er_image).astype(int)
        
    pixels = np.count_nonzero(er_image)
    
    return pixels, filled_image

def point_in_hull(point, hull, tolerance=1e-12):
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)


def get_surfaces(img, header):
    # Determine pixel size
    sx, sy, _ = header.get_zooms()
    pixel_area = sx * sy

    # Prepare empty arrays
    surfaces1 = np.ones(img.shape) * -1
    surfaces2 = np.ones(img.shape) * -1
    surfaces1_mm2 = np.ones(img.shape[2]) * -1
    surfaces2_mm2 = np.ones(img.shape[2]) * -1

    # Loop over every slice    
    for slice in tqdm(range(img.shape[2])):
        image = img[:,:,slice]
        
        try:
            # Isolate spin from image
            st_image, cof = isolate_spine(image)

            # Determine surface 2
            surface2_pixels, spine_image = spine_surface(st_image)
            surface2_mm2 = surface2_pixels * pixel_area

            # Determine total surface
            contours = intensity_seg(image, min=-1000, max=-300) # Hounsfields units
            contours = find_lungs_adapted(contours)
            contours_total = np.concatenate((contours[0], contours[1]))
            hull = ConvexHull(contours_total)
            surface12 = round(hull.volume)

            # Determine surface 1 (nr of pixels and area)
            surface1_pixels = surface12 - surface2_pixels
            surface1_mm2 = surface1_pixels * pixel_area

            x_coords = np.array(contours_total[hull.vertices,0],int)
            y_coords = np.array(contours_total[hull.vertices,1],int)

            # Rotate the coordinates for the total area with 90 degrees 
            # to correspond to the orientation of the original image
            xy = np.column_stack((x_coords,y_coords))
            xy_rotated = np.rot90(xy)

            # Determine surface 1
            x_dim = image.shape[0]
            y_dim = image.shape[1]
            surf1 = np.full((x_dim, y_dim), False)
            for x_cor in range(x_dim):
                for y_cor in range(y_dim):
                    if point_in_hull([x_cor,y_cor],hull)==True and spine_image[x_cor,y_cor]==False:
                        surf1[x_cor,y_cor]=True

            surface1 = remove_small_objects(surf1, min_size=1000)

            # Put in pre-defined arrays
            surfaces1[:,:,slice] = surface1
            surfaces2[:,:,slice] = spine_image
            surfaces1_mm2[slice] = surface1_mm2
            surfaces2_mm2[slice] = surface2_mm2
            
        except:
            pass
    
    return surfaces1, surfaces2, surface1_mm2, surface2_mm2 
