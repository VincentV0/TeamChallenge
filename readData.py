import numpy as np
from xml.dom import minidom
import nibabel as nib


def read_nii(path, rotate=True):
    """
    Loads the NIFTI images.
    """
    ct_input = nib.load(path)
    ct_img = ct_input.get_fdata()
    ct_img_header = ct_input.header
    if rotate:
        ct_img = np.rot90(ct_img)
    return ct_img, ct_img_header

def read_nii_masks(path1, path2, rotate=True):
    """
    Loads the masks resulting from the CSO files.
    """
    mask1,_ = read_nii(path1, rotate)
    mask2,_ = read_nii(path2, rotate)
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