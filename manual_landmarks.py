
import numpy as np
import matplotlib.pyplot as plt
from ScrollPlot import ScrollPlot

def get_manual_landmarks(img_data, title=''):
    """
    Get a single point from an image
    """
    fig, ax = plt.subplots(1, 1)
    sp = ScrollPlot(ax, img_data, None, close_on_click=True, ax_title=title)
    fig.canvas.mpl_connect('scroll_event', sp.on_scroll)
    fig.canvas.mpl_connect('button_press_event', sp.on_click)
    plt.show()
    return sp.get_marked_points()

def manual_translation(preop_data, postop_data, landmarks):
    # Get a single point as a landmark for the first and second image
    preop_lm = get_manual_landmarks(preop_data, "Select a single landmark for image 1")[0]
    postop_lm = get_manual_landmarks(postop_data, "Select a single landmark for image 2")[0]

    t_data = postop_data
    min_value = np.amin(postop_data)

    # Get differences of each axis
    x_diff = preop_lm[0] - postop_lm[0]
    y_diff = preop_lm[1] - postop_lm[1]
    slice_diff = preop_lm[2] - postop_lm[2]

    # Slice translation
    t_data = np.roll(t_data, slice_diff, axis=2)
    if slice_diff < 0:
        t_data[:,:,slice_diff:] = min_value
    else:
        t_data[:,:,:slice_diff] = min_value
            
    # Row and column translation
    for slice in range(t_data.shape[2]):
        t_data[:,:,slice] = np.roll(t_data[:,:,slice], x_diff, axis=0)
        t_data[:,:,slice] = np.roll(t_data[:,:,slice], y_diff, axis=1)
        
        if x_diff < 0:
            t_data[x_diff:,:,slice] = min_value
        else:
            t_data[:x_diff,:,slice] = min_value

        if y_diff < 0:
            t_data[:,y_diff:,slice] = min_value
        else:
            t_data[:,:y_diff,slice] = min_value
    
    # Translate the landmarks to their new positions
    landmarks[:,0] += y_diff
    landmarks[:,1] += x_diff
    landmarks[:,2] += slice_diff
        
    return t_data, landmarks