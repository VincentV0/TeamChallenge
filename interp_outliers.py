# reference_slice=true_markers[lm-1][2], threshold=50 
import pandas as pd
import math
from scipy.spatial import distance
import numpy as np

def find_nearest(array,value, option):
    idx = np.searchsorted(array, value, side=option)
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]
    

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
                
        #box = np.ones(3)/3
        #x_smooth = np.convolve(x1, box, mode='same')
        #y_smooth = np.convolve(y1, box, mode='same')
        x[start_point:end_point] = x1 #x_smooth
        y[start_point:end_point] = y1 #y_smooth

        x[x == -1] = -1
        x[y == -1] = -1

    return x,y


def interpolate(x_list, y_list):
    complete_list_x=np.zeros(len(x_list))
    nonzeros_x=np.nonzero(x_list)
    for i in range(len(x_list)):
        if x_list[i]==0:
            nearest_right=find_nearest(nonzeros_x[0],i, "right")
            nearest_left=find_nearest(nonzeros_x[0],i, "left")
            if i ==(len(x_list)-1):
                complete_list_x[i]=x_list[nearest_left]
            elif i>1 :
                complete_list_x[i]=(x_list[nearest_left]+x_list[nearest_right])/2
            else:
                complete_list_x[i]=x_list[nearest_right]
        else:
            complete_list_x[i]=x_list[i] 
        
    complete_list_y=np.zeros(len(y_list))
    nonzeros_y=np.nonzero(y_list)
    for i in range(len(y_list)):
        if y_list[i]==0:
            nearest_right=find_nearest(nonzeros_y[0],i, "right")
            nearest_left=find_nearest(nonzeros_y[0],i, "left")
            if i ==(len(y_list)-1):
                complete_list_y[i]=y_list[nearest_left]
            elif i>1 :
                complete_list_y[i]=(y_list[nearest_left]+y_list[nearest_right])/2
            else:
                complete_list_y[i]=y_list[nearest_right]
        else:
            complete_list_y[i]=y_list[i]
    return complete_list_x, complete_list_y

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