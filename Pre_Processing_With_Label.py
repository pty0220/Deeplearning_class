import os
import glob
import pandas as pd
import numpy as np


def pre_process(dir, label):
    dir = dir #'C:/Users/FUS/Desktop/code/Data/Data/Train/PD'
    dir_all = glob.glob(dir+"/*.CSV")

    data = []
    for i in dir_all :
        temp = pd.read_csv(i)
        data.append(temp.iloc[:,-1].values)

    data = np.array(data)
    data_size = data.shape
    data_label = np.ones((data_size[0], data_size[1]+1))
    data_label = data_label*label
    data_label[:,:-1] = data

    return data_label

PD_label = pre_process('C:/Users/FUS/Desktop/code/Data/Train/PD', 0)
Noise_label = pre_process('C:/Users/FUS/Desktop/code/Data/Train/Noise', 1)
Unknown_label = pre_process('C:/Users/FUS/Desktop/code/Data/Train/Unknown', 2)

train_data = np.concatenate((PD_label, Noise_label, Unknown_label), axis =0)

DATA = pd.DataFrame(data = train_data)
DATA.to_csv("../pre_data/PD_Noise_Unknown.csv", index = False, header = False)


