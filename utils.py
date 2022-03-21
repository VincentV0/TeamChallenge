import math
import numpy as np

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

def haller_index(lms):
    """
    
    Haller index is defined by the width divided by length of the thorax. 
    therefor points p2, p4 (for A-P length) and p1 and p3 (for the ML length) is needed. 
    
    !Make sure the scans are oriented with the dorsal side at the bottom of the image!

    ** does not work yet, because we do not have p4 yet! **
    """
    
    AP_length = abs(lms[1][:,1]-lms[3][:,1]) # anterior, posterior length
    ML_length = abs(lms[2][:,0]-lms[4][:,0]) # medial, lateral length
    
    HI_index = ML_length/AP_length # Haller index
    return HI_index

def sagittal_diameter(p2, anterior_foramen_point):
    
    """
    calculating the sagittal diameter by taking the length between point 1 and the anterior point of the spinal foramen.
    
    ** does not work yet, because we do not have the anterior foramen point yet! **
    """
    sagit_dia = abs(p2[0]-anterior_foramen_point[0])
    
    #return sagit_dia
    
    