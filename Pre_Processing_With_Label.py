import os
import glob
import pandas as pd
import numpy as np


def pre_process(dir, label):
    dir = dir #'C:/Users/FUS/Desktop/code/Data/Data/Train/PD'
    dir_all = glob.glob(dir+"/*.CSV")

    data = []
    for i in dir_all :
        temp = pd.read_csv(i, engine='python')
        data.append(temp.iloc[:,-1].values)
        print('processing num: ',i)

    data = np.array(data)
    data_size = data.shape
    data_label = np.ones((data_size[0], data_size[1]+1))
    data_label = data_label*label
    data_label[:,:-1] = data

    return data_label

# PD_label = pre_process(r'C:\Users\User\Desktop\Deep_learning\Data\마화CH3\마화_CH3_PD', 0)
# Noise_label = pre_process(r'C:\Users\User\Desktop\Deep_learning\Data\마화CH3\마화_CH3_Noise', 1)
# Unknown_label = pre_process('C:/Users/FUS/Desktop/code/Data/Train/Unknown', 2)

whole = pd.read_csv('../pre_data/PD_Noise_Unknown_labeled.csv')
CH1 = pd.read_csv('../pre_data/마화1.csv', engine='python')
CH2 = pd.read_csv('../pre_data/마화2.csv', engine='python')
CH3 = pd.read_csv('../pre_data/마화3.csv', engine='python')

# whole = whole.iloc[:,:].values
# CH1 = CH1.iloc[:,:].values
# CH2 = CH2.iloc[:,:].values
# CH3 = CH3.iloc[:,:].values
# train_data = np.concatenate((whole, CH1, CH2, CH3), axis =0)



train_data = np.concatenate((whole, CH1, CH2, CH3), axis =0)
DATA = pd.DataFrame(data = train_data)


DATA.to_csv("../pre_data/whole_마화123.csv", index = False, header = False)


