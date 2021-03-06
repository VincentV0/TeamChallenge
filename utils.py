import math
import numpy as np
import pandas as pd

def NormalizeData(data):
    """ Normalizes data to the [0, 255] range; not required right now """
    return (data - np.min(data)) / (np.max(data) - np.min(data)) * 255

def rotate(point, origin, angle=-math.pi/2):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def rotate_landmarks(landmarks, origin, angle=-math.pi/2):
    """
    Rotate array of points to new frame of reference using origin and angle.
    """
    for i in range(len(landmarks)):
        x = landmarks[i,0]
        y = landmarks[i,1]
        
        if x != -1 and y != -1:
            landmarks[i][0] = np.array(rotate((x,y), origin, angle)[0])
            landmarks[i][1] = np.array(rotate((x,y), origin, angle)[1])
    return landmarks

def haller_index(lms, nii_header):
    """    
    Haller index is defined by the width divided by length of the thorax. 
    therefor points p2, p4 (for A-P length) and p1 and p3 (for the ML length) is needed. 
    
    !Make sure the scans are oriented with the dorsal side at the bottom of the image!
    """
    HI = np.ones(lms[1].shape[0]) * -1
    for slice in range(lms[1].shape[0]):

        # All points have to be known to calculate the Haller index
        if -1 in lms[1][slice] or -1 in lms[2][slice] \
            or -1 in lms[3][slice] or -1 in lms[4][slice]: continue
        else:
            # Calculate Anterior-Posterior length in voxels and convert to length in meters
            AP_length = abs(lms[1][slice,1]-lms[3][slice,1]) # anterior, posterior length
            _, AP_len_meter, _ = convert_units_distance(0, AP_length, 0, nii_header)

            # Calculate Median-Lateral length in voxels and convert to length in meters
            ML_length = abs(lms[2][slice,0]-lms[4][slice,0]) # medial, lateral length
            ML_len_meter, _, _ = convert_units_distance(ML_length, 0, 0, nii_header)

            # Calculate Haller index
            HI_index = ML_len_meter/AP_len_meter
            HI[slice] = HI_index
    return HI


def convert_units_distance(x_val, y_val, z_val, nii_header):
    """
    Takes a distance in 'pixels' and converts this to meters [?]. 
    Note: only distances, not absolute positions (there is no clear origin)!
    Code based on information in https://brainder.org/2012/09/23/the-nifti-file-format/
    Returns distances in meters.
    """
    # Shape of the voxels
    qfac, sx, sy, sz, _, _, _, _ = nii_header['dim'] = nii_header['pixdim']
    
    # Nr of pixels in each direction
    ndim, nx, ny, nz, _, _, _, _ = nii_header['dim']

    # Find units (https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/xyzt_units.html)
    unit_id = nii_header['xyzt_units']
    temporal_id = unit_id // 8
    spatial_id = unit_id - temporal_id*8
    
    if spatial_id == 1: # unit = meters; don't do anything
        pass
    if spatial_id == 2: # unit = millimeters; compensate with factor 1e-3
        sx, sy, sz = sx*1e-3, sy*1e-3, sz*1e-3
    if spatial_id == 3: # unit = micrometers; compensate with factor 1e-6
        sx, sy, sz = sx*1e-6, sy*1e-6, sz*1e-6

    # Calculate distance in meters
    x_m = x_val * sx
    y_m = y_val * sy
    z_m = z_val * sz * qfac

    return x_m, y_m, z_m



def export_to_excel(landmarks, filename):
    """
    Export the dictionary of landmarks to an xlsx-file.
    Structure of xlsx-file:
    1_x | 1_y | 1_z | 2_x | 2_y | ...
    """
    df = pd.DataFrame()
    for lm in landmarks:
        landmark = landmarks[lm]
        df[f"{lm}_x"] = landmark[:,0]
        df[f"{lm}_y"] = landmark[:,1]
        df[f"{lm}_z"] = np.arange(landmark.shape[0])
    df.to_excel(filename)


def haller_index_export(HI1, HI2=None, filename=''):
    """
    Export the array with Haller indices to an xlsx-file.
    Structure of xlsx-file:
    slice | HI1 | (HI2)
    """
    df = pd.DataFrame()
    df[f"HI1"] = HI1
    if HI2 is not None:
        df[f"HI2"] = HI2
    df.to_excel(filename)
