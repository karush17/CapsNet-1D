"""Preprocesses and cleans the gesture activity dataset."""

import csv
import io
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import medfilt

path_label1 = r'C:\Users\Karush\.spyder-py3\subj_11'
path_label2 = r'C:\Users\Karush\.spyder-py3'
fio = io.open(path_label2+r'\acc_par.csv','rt')
cal_acc = csv.reader(fio)
cal_acc = list(cal_acc)
cal_acc = np.array(cal_acc).astype(np.float)

fio = io.open(path_label2+r'\cal_gyro.csv','rt')
cal_gyro = csv.reader(fio)
cal_gyro = list(cal_gyro)
cal_gyro = np.array(cal_gyro).astype(np.float)

def get_angles(X: np.ndarray) -> np.ndarray:
    """Returns the X, Y and Z angles."""    
    angles = np.zeros((0,3))
    alpha = 0.85
    dt = 0.010
    for w in range(0,len(X)):
        angle_conc = np.zeros((0,0))
        acc_angle_y = np.arctan(-1*X[w,0]/(np.sqrt(X[w,1]**2 + X[w,2]**2)))
        acc_angle_x = np.arctan(-1*X[w,1]/(np.sqrt(X[w,0]**2 + X[w,2]**2)))
        acc_angle_z = 0

        if (w==0):
            last_angle_x = 0
            last_angle_y = 0
            last_angle_z = 0

        gyro_angle_x = X[w,3]*dt + last_angle_x
        gyro_angle_y = X[w,4]*dt + last_angle_y
        gyro_angle_z = X[w,5]*dt + last_angle_z

        angle_x = alpha*gyro_angle_x + (1-alpha)*acc_angle_x
        angle_y = alpha*gyro_angle_y + (1-alpha)*acc_angle_y
        angle_z = alpha*gyro_angle_z + (1-alpha)*acc_angle_z

        last_angle_x = angle_x
        last_angle_y = angle_y
        last_angle_z = angle_z

        angle_conc = [angle_x,angle_y,angle_z]
        angles = np.r_['0,2',angles,angle_conc]

    return angles


for act_num in range(1,21):
    fio = io.open(path_label1 + r'\act_' + str(act_num) + '.csv','rt')
    data = csv.reader(fio)
    data = list(data)
    data = np.array(data).astype(np.float)
    indx = []

    ##Correct for unwanted timestamps
    for j in range(0,len(data)):
        if (data[j,0]>=10000 or data[j,0]==0):
            data[j,:] = 0

    ##Correct for future timestamps
    for l in range(1,5):        
        for k in range(1,len(data)):        
            if (data[k-1,0]>data[k,0]):
                if (data[k-1,0]>=9990):
                    data[k-1,:] = data[k-1,:]
                else: 
                    data[k-1,:] = 0

    ##Delete nan rows
    data= np.delete(data,np.where(~data.any(axis=1))[0], axis=0)

    ##Index each signal
    for j in range(0,len(data)):
        if (data[j,0]>=9990):
            indx = np.append(indx,j)

    del_idx = []
    ##Correct for indices
    for i in range(1,len(indx)):
        if (indx[i]-indx[i-1]<500):
            indx[i-1] = 0
            del_idx = np.append(del_idx,i-1)
    indx = np.delete(indx,del_idx, axis=0)
    indx = np.insert(indx,0,0)

    ##Calibration
    data_new = np.zeros((0,10))
    for k in range(0,10):
        rep1 = k
        rep2 = k+1

        ## Accelerometer calibration
        time = data[int(indx[rep1]+1):int(indx[rep2]),0]
        ones_vec = np.ones((len(time),1))
        acc_data = data[int(indx[rep1]+1):int(indx[rep2]),1:4]
        acc_data = np.c_[acc_data,ones_vec]
        mod_acc = np.matmul(acc_data,cal_acc)
        data[int(indx[rep1]+1):int(indx[rep2]),1:4] = mod_acc

        ## Gyroscope calibration
        gyro_data = data[int(indx[rep1]+1):int(indx[rep2]),4:7]
        gyro_data[:,0] = (gyro_data[:,0]- cal_gyro[0,0])*cal_gyro[0,1]
        gyro_data[:,1] = (gyro_data[:,1]- cal_gyro[1,0])*cal_gyro[1,1]
        gyro_data[:,2] = (gyro_data[:,2]- cal_gyro[2,0])*cal_gyro[2,1]
        data[int(indx[rep1]+1):int(indx[rep2]),4:7] = gyro_data

        ## Angle estimation
        angles = get_angles(np.c_[mod_acc,gyro_data])
        data_temp = np.c_[time,mod_acc,gyro_data,angles]
        data_new = np.r_['0,2',data_new,data_temp]

    ##Peak correction using median filter
    data = data_new
    for k in range(1,len(data[1,:])):
        data[:,k] = medfilt(data[:,k])
        data[:,k] = -data[:,k]
        data[:,k] = medfilt(data[:,k])
        data[:,k] = -data[:,k]

    np.savetxt("act_"+str(act_num)+".csv",data,delimiter=",")
