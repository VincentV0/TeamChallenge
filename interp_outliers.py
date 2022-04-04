from scipy.spatial import distance
import numpy as np

def interpol_alt(x,y):
    # Determine which slices have a prediction
    t_true = np.where(x != -1)[0]

    if len(t_true) > 0:
        x_true = x[t_true]
        y_true = y[t_true]

        # Skip the very bottom and very top of the image; we do not extrapolate
        start_point = np.min(t_true)
        end_point = np.max(t_true)

        # Define the range of slices on which to interpolate
        t_interpol = np.arange(start_point, end_point)

        # Perform the interpolation and save to array
        x1 = np.interp(t_interpol, t_true, x_true)
        y1 = np.interp(t_interpol, t_true, y_true)
        x[start_point:end_point] = np.round(x1) 
        y[start_point:end_point] = np.round(y1)

    return x,y

def find_outliers(x_list, y_list, reference_slice, threshold):
    # Define array for distances
    dist=np.zeros(len(x_list))

    # Work from the reference slice upwards
    for i in range(reference_slice, len(x_list)):
        m=1
        while x_list[i-m]==-1:
            m+=1
        else:
            # Calculate previous and current point
            previous_point=(x_list[i-m], y_list[i-m])
            point=(x_list[i], y_list[i])

            # Calculate distance; if this distance is too high, drop this point
            dist=distance.euclidean(previous_point,point)
            if dist>threshold:
                x_list[i]=-1
                y_list[i]=-1          

    # Work from the reference slide downwards
    for i in range(reference_slice-2,-1,-1):
        m=1
        while x_list[i+m]==-1:
            m+=1
        else:
            # Calculate previous and current point
            previous_point=(x_list[i+1], y_list[i+1])
            point=(x_list[i], y_list[i])

            # Calculate distance; if this distance is too high, drop this point
            dist=distance.euclidean(previous_point,point)
            if dist>threshold:
                x_list[i]=-1
                y_list[i]=-1
    
    # Interpolate the points that have just been dropped
    interpolation=interpol_alt(x_list, y_list)
    final_x=interpolation[0]
    final_y=interpolation[1]
    return final_x,final_y