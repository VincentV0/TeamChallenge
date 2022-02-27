import numpy as np
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
from xml.dom import minidom
import nibabel as nib



DATA_PATH = r"C:\\temp\\TC_DATA\\"
DATA_PATH_SCOLIOTIC = os.path.join(DATA_PATH, "Scoliose")
DATA_PATH_NONSCOLIOTIC = os.path.join(DATA_PATH, "Nonscoliotic")



def read_nii(path):
    """
    Loads the NIFTI images.
    """
    img = sitk.ReadImage(path)
    img_arr = np.moveaxis(sitk.GetArrayFromImage(img).astype('float'), 0, -1)
    return img_arr


def read_nii_masks(path1, path2):
    """
    Loads the masks resulting from the CSO files.
    """
    mask1 = read_nii(path1)
    mask2 = read_nii(path2)
    return mask1, mask2


def read_xml(path):
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
        markerlist.append(tuple(coords[0:3]))
    
    return np.array(markerlist)