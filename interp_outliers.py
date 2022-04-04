from scipy.spatial import distance
import numpy as np

def interpol_alt(x,y):
    t_true = np.where(x != -1)[0]
    if len(t_true) > 0:
        x_true = x[t_true]
        y_true = y[t_true]

        start_point = np.min(t_true)
        end_point = np.max(t_true)

        t_interpol = np.arange(start_point, end_point)

        x1 = np.interp(t_interpol, t_true, x_true)
        y1 = np.interp(t_interpol, t_true, y_true)
                

        x[start_point:end_point] = np.round(x1) #x_smooth
        y[start_point:end_point] = np.round(y1) #y_smooth

        x[x == -1] = -1
        x[y == -1] = -1

    return x,y

def find_outliers(x_list, y_list, reference_slice, threshold):
    dist=np.zeros(len(x_list))
    for i in range(reference_slice, len(x_list)):
        m=1
        while x_list[i-m]==-1:
            m+=1
        else:
            previous_point=(x_list[i-m], y_list[i-m])
            point=(x_list[i], y_list[i])
            dist=distance.euclidean(previous_point,point)
            if dist>threshold:
                x_list[i]=-1
                y_list[i]=-1          
    for i in range(reference_slice-2,-1,-1):
        m=1
        while x_list[i+m]==-1:
            m+=1
        else:
            previous_point=(x_list[i+1], y_list[i+1])
            point=(x_list[i], y_list[i])
            dist=distance.euclidean(previous_point,point)
            if dist>threshold:
                x_list[i]=-1
                y_list[i]=-1
    interpolation=interpol_alt(x_list, y_list)
    final_x=interpolation[0]
    final_y=interpolation[1]
    return final_x,final_y