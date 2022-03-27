import numpy as np
#import pandas as pd
import math
from scipy.spatial import distance

def find_nearest(array,value, option):
    idx = np.searchsorted(array, value, side=option)
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]
    
def interpolate(x_list, y_list):
    complete_list_x=np.zeros(len(x_list))
    #nonzeros_x=np.nonzero(x_list)
    nonzeros_x=np.where(x_list == -1)
    for i in range(len(x_list)):
        if x_list[i]==-1:
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
    #nonzeros_y=np.nonzero(y_list)
    nonzeros_y=np.where(y_list == -1)

    for i in range(len(y_list)):
        if y_list[i]==-1:
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

def find_outliers(x_list, y_list, reference):
    index_outliers=[]
    final_x = np.array([])
    final_y = np.array([])
    while not np.array_equal(final_x, x_list) and not np.array_equal(final_y, y_list):
        dist=np.zeros(len(x_list))
        for i in range(len(x_list)):
            if i>1:
                point=(x_list[i], y_list[i])
                dist[i]=distance.euclidean(point,reference)
        mean_distance=np.mean(dist)
        sd=np.std(dist)
        for x in range(len(x_list)):
            if dist[x]<(mean_distance-2*sd):
                x_list[x]=-1
                y_list[x]=-1
                index_outliers.append(x)
            elif dist[x]>(mean_distance+2*sd):
                index_outliers.append(x)
                x_list[x]=-1
                y_list[x]=-1
        interpolation=interpolate(x_list,y_list)
        final_x=interpolation[0]
        final_y=interpolation[1]
    return final_x,final_y, index_outliers
