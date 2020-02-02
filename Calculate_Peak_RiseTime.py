import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from scipy.signal import find_peaks
from scipy.misc import electrocardiogram
from sklearn.preprocessing import normalize

l2n = lambda l: np.array(l)
n2l = lambda n: list(n)

# Import and split data
data = pd.read_csv('../pre_data/PD_Noise_Unknown_labeled.csv', header = None)
data = data.sort_values(512, ascending=[True])

time = pd.read_csv('../pre_data/Times.csv', header = None)

"""Model Building"""
# Split features and labels
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
time = time.iloc[:,:].values


peak = x.max(axis=1)
riseTime = np.squeeze(time[x.argmax(axis=1)])

changeIdx = np.min(np.where(y==1))

peakPD = peak[:changeIdx]
riseTimePD = riseTime[:changeIdx]
peakNoise = peak[changeIdx:]
riseTimeNoise = riseTime[changeIdx:]

plt.scatter(peakNoise, riseTimeNoise, marker='x', color='blue')
plt.scatter(peakPD, riseTimePD, marker='.', color='red')

plt.xlabel('Peak [mV]')
plt.ylabel('riseTime [us]')
plt.show()




a=1