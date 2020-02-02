import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

rand_idx = [1588, 224, 436, 380]

def result_show(idx):

    if idx == 2:
        result_text = 'Unknown'
    if idx == 1:
        result_text = 'Noise'
    if idx == 0:
        result_text = 'PD'

    return result_text




data = pd.read_csv('../pre_data/PD_Noise_Unknown.csv')
time = pd.read_csv('../pre_data/Times.csv')

# Split features and labels
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

time = time.iloc[:,:].values
time = np.squeeze(time)


fig = plt.figure()

num_plt = 1
for i in rand_idx:

    ax = fig.add_subplot(len(rand_idx), 1, num_plt)
    ax.plot(time, x[i, :-1])


    real_result = result_show(y[i])

    plt.xlabel('time')
    plt.ylabel('voltage (mV)')
    #plt.xlim((0,3000))
    #plt.ylim((-200,200))
    plt.title('Label: '+real_result)
    ax.set_aspect('auto')
    plt.tight_layout()  # not strictly part of the question
    num_plt = num_plt+1
#plt.tight_layout()
plt.show()
