import os
import glob
import pandas as pd
import numpy as np
import re

def pre_process_with_PHMV(dir):

    dir_all = glob.glob(dir+"/*.CSV")
    length = len(dir_all)

    data = []
    PHMV = np.ones((length,2))
    i = 0

    for dirOne in dir_all:

        numbers = re.findall(r'\d+', dirOne)
        PHMV[i,0] = float(numbers[3]) + float(numbers[4])/10
        PHMV[i,1] = float(numbers[5]) + float(numbers[6])/10

        temp = pd.read_csv(dirOne)
        data.append(temp.iloc[:,-1].values)
        i = i+1
        print(i)

    data = np.array(data)

    return data, PHMV

CH1, PHMV = pre_process_with_PHMV('/Users/home/Desktop/Deep_learning/Data/CH1_CSV')


DATA = pd.DataFrame(data = CH1)
DATA.to_csv("../pre_data/CH1_whole_data.csv", index = False, header = False)

DATA = pd.DataFrame(data = PHMV)
DATA.to_csv("../pre_data/CH1_PHMV.csv", index = False, header = False)

