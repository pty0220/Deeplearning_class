import os
import glob
import pandas as pd
import numpy as np

PD_dir = 'C:/Users/FUS/Desktop/code/Data/Test/PD'
PD_dir_all = glob.glob(PD_dir+"/*.CSV")

PD = []
for i in PD_dir_all :
    temp = pd.read_csv(i)
    PD.append(temp.iloc[:,-1].values)

PD = np.array(PD)
data_size = PD.shape
PD_label = np.zeros((data_size[0], data_size[1]+1))
PD_label[:,:-1] = PD


noise_dir = 'C:/Users/FUS/Desktop/code/Data/Test/Noise'
noise_dir_all = glob.glob(noise_dir+"/*.CSV")

noise = []
for i in noise_dir_all :
    temp = pd.read_csv(i)
    noise.append(temp.iloc[:,-1].values)

noise = np.array(noise)
data_size = noise.shape
noise_label = np.ones((data_size[0], data_size[1]+1))
noise_label[:,:-1] = noise


test_data = np.concatenate((PD_label, noise_label), axis =0)
#np.savetxt("PD_noise_data.csv", train_data, delimiter=" ")

DATA = pd.DataFrame(data = test_data)
DATA.to_csv("../pre_data/PD_noise_test_data.csv", index = False, header = False)

times = temp.iloc[:,0].values
#np.savetxt("Times.csv", times, delimiter=" ")

DATA_t = pd.DataFrame(data = times)
DATA_t.to_csv("../pre_data/Times.csv", index = False, header = False)


####




PD_dir = 'C:/Users/FUS/Desktop/code/Data/Train/PD'
PD_dir_all = glob.glob(PD_dir+"/*.CSV")

PD = []
for i in PD_dir_all :
    temp = pd.read_csv(i)
    PD.append(temp.iloc[:,-1].values)

PD = np.array(PD)
data_size = PD.shape
PD_label = np.zeros((data_size[0], data_size[1]+1))
PD_label[:,:-1] = PD


noise_dir = 'C:/Users/FUS/Desktop/code/Data/Train/Noise'
noise_dir_all = glob.glob(noise_dir+"/*.CSV")

noise = []
for i in noise_dir_all :
    temp = pd.read_csv(i)
    noise.append(temp.iloc[:,-1].values)

noise = np.array(noise)
data_size = noise.shape
noise_label = np.ones((data_size[0], data_size[1]+1))
noise_label[:,:-1] = noise


train_data = np.concatenate((PD_label, noise_label), axis =0)
#np.savetxt("PD_noise_data.csv", train_data, delimiter=" ")

DATA = pd.DataFrame(data = train_data)
DATA.to_csv("../pre_data/PD_noise_data.csv", index = False, header = False)

times = temp.iloc[:,0].values
#np.savetxt("Times.csv", times, delimiter=" ")

DATA_t = pd.DataFrame(data = times)
DATA_t.to_csv("../pre_data/Times.csv", index = False, header = False)



whole_data = np.concatenate((train_data, test_data), axis =0)
DATA = pd.DataFrame(data = whole_data)
DATA.to_csv("../pre_data/whole_PD_noise_data.csv", index = False, header = False)



a = 1