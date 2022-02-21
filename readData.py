import numpy as np
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
from xml.dom import minidom


DATA_PATH = r"C:\\temp\\TC_DATA\\"
DATA_PATH_SCOLIOTIC = os.path.join(DATA_PATH, "Scoliose")
DATA_PATH_NONSCOLIOTIC = os.path.join(DATA_PATH, "Nonscoliotic")



def read_nii(path):
    """
    Loads the NIFTI images.
    """
    img = sitk.ReadImage(path)
    img_arr = sitk.GetArrayFromImage(img).astype('float')
    return img_arr


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
        coords = [float(c) for c in coords.split(' ')]
        markerlist.append(tuple(coords))
    
    return markerlist




# Load both files and print marker list + image shape
file = "2postop"
filepath = os.path.join(DATA_PATH_SCOLIOTIC, file)
img = read_nii(filepath + ".nii")
markers = read_xml(filepath + ".xml")
print("\nMARKERS: ")
for x in markers: print(x)
print("\nIMAGE SHAPE:  ", img.shape)
