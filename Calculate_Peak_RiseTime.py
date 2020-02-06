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

plt.figure(figsize=(10, 8), facecolor='white')
plt.scatter(peakNoise, riseTimeNoise, marker='x', color='blue', label = "Noise signal")
plt.scatter(peakPD, riseTimePD, marker='.', color='red', label = "PD signal")
plt.legend(loc='upper left')
plt.xlabel('Peak [mV]')
plt.ylabel('Rise Time [us]')


noiseSample = range(peakNoise.shape[0])
pdSample = range(peakPD.shape[0])

plt.figure(figsize=(10, 8), facecolor='white')
# plt.stem(peakNoise, noiseSample, linefmt='blue')
# plt.stem(peakPD, pdSample, linefmt='red')
plt.scatter(peakNoise, noiseSample, marker ='x', color='blue', label = 'Noise signal')
plt.scatter(peakPD, pdSample, marker ='.',color='red', label = "PD signal")
plt.legend(loc='upper left')
plt.xlabel('Sampling')
plt.ylabel('Peak [mV]')


plt.figure(figsize=(10, 8), facecolor='white')
# plt.stem(peakNoise, noiseSample, linefmt='blue')
# plt.stem(peakPD, pdSample, linefmt='red')
plt.scatter(riseTimeNoise, noiseSample, marker ='x', color='blue', label = 'Noise signal')
plt.scatter(riseTimePD, pdSample, marker ='.',color='red', label = "PD signal")
plt.legend(loc='upper left')
plt.xlabel('Sampling')
plt.ylabel('Rise time [mV]')
plt.show()

a=1