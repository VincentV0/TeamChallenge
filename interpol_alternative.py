from scipy.ndimage import gaussian_filter1d
import numpy as np



def interpol_alt(marks):
    x = marks[:,0]
    y = marks[:,1]

    t_true = np.where(x != -1)[0]
    x_true = x[t_true]
    y_true = y[t_true]

    start_point = np.min(t_true)
    end_point = np.max(t_true)

    t_interpol = np.arange(start_point, end_point)

    x1 = np.interp(t_interpol, t_true, x_true)
    y1 = np.interp(t_interpol, t_true, y_true)
            
    box = np.ones(3)/3
    x_smooth = np.convolve(x1, box, mode='same')
    y_smooth = np.convolve(y1, box, mode='same')
    x[start_point:end_point] = x_smooth
    y[start_point:end_point] = y_smooth

    x[x == -1] = -1
    x[y == -1] = -1

    marks[:,0] = x
    marks[:,1] = y
    return marks

