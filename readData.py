import numpy as np
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
from xml.dom import minidom
import nibabel as nib



DATA_PATH = r"C:\\temp\\TC_DATA\\"
DATA_PATH_SCOLIOTIC = os.path.join(DATA_PATH, "Scoliose")
DATA_PATH_NONSCOLIOTIC = os.path.join(DATA_PATH, "Nonscoliotic")



def read_nii(path, rotate=True):
    """
    Loads the NIFTI images.
    """
    ct_input = nib.load(path)
    ct_img = ct_input.get_fdata()
    if rotate:
        ct_img = np.rot90(ct_img)
    return ct_img

def read_nii_masks(path1, path2, rotate=True):
    """
    Loads the masks resulting from the CSO files.
    """
    mask1 = read_nii(path1, rotate)
    mask2 = read_nii(path2, rotate)
    return mask1, mask2


def read_xml(path, rotate=True):
    """
    Reads a list of markers from the XML file.
    """
    # Load XML
    file = minidom.parse(path)

    #use getElementsByTagName() to get tag
    markers = file.getElementsByTagName('Item')

    # Save positions to marker list
    markerlist = []
    for elem in markers:
        positions = elem.getElementsByTagName('pos')
        coords = positions[0].firstChild.nodeValue
        coords = [round(float(c)) for c in coords.split(' ')]
        if rotate:
            markerlist.append((coords[1], coords[0], coords[2]))
        else:
            markerlist.append(tuple(coords[0:3]))
    
    return np.array(markerlist)