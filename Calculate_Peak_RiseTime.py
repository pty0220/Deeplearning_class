import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from scipy.signal import find_peaks
from scipy.misc import electrocardiogram
from sklearn.preprocessing import normalize
from scipy.signal import find_peaks

l2n = lambda l: np.array(l)
n2l = lambda n: list(n)

# Import and split data
dataPath = '../pre_data/whole_마화123.csv'

data = pd.read_csv(dataPath, header = None, engine='python')
data = data.sort_values(512, ascending=[True])

time = pd.read_csv('../pre_data/Times.csv', header = None)

"""Model Building"""
# Split features and labels
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
time = time.iloc[:,:].values


maximum = x.max(axis=1)
changeIdx = np.min(np.where(y==1))

firstPeakidx = []
peak = []
peakPD =[]
riseTimePD =[]
peakNoise =[]
riseTimeNoise =[]

for i in range(len(x)):
    signal = x[i,:]

    peaks, _ = find_peaks(signal, height= maximum[i]*0.1)
    firstPeakidx.append(peaks[0])
    peak.append(signal[peaks[0]])
    # plt.plot(signal)
    # plt.plot(firstPeakidx[i], signal[firstPeakidx[i]], "x")
    # plt.plot(np.zeros_like(signal), "--", color="gray")
    # plt.show()


firstPeakidx = np.array(firstPeakidx)
peak = np.array(peak)
riseTime = np.squeeze(time[firstPeakidx])


for i in range(len(x)):
    if y[i] == 0:
        peakPD.append(peak[i])
        riseTimePD.append(riseTime[i])
    else:
        peakNoise.append(peak[i])
        riseTimeNoise.append(riseTime[i])

peakPD = np.array(peakPD)
riseTimePD = np.array(riseTimePD)
peakNoise = np.array(peakNoise)
riseTimeNoise = np.array(riseTimeNoise)



plt.figure(figsize=(10, 8), facecolor='white')
plt.scatter(peakNoise, riseTimeNoise, marker='x', color='blue', label = "Noise signal")
plt.scatter(peakPD, riseTimePD, marker='.', color='red', label = "PD signal")
plt.legend(loc='upper left')
plt.xlabel('Peak [mV]')
plt.ylabel('Rise Time [us]')
plt.title(dataPath)

noiseSample = range(peakNoise.shape[0])
pdSample = range(peakPD.shape[0])

plt.figure(figsize=(10, 8), facecolor='white')
# plt.stem(peakNoise, noiseSample, linefmt='blue')
# plt.stem(peakPD, pdSample, linefmt='red')
plt.scatter(noiseSample, peakNoise, marker ='x', color='blue', label = 'Noise signal')
plt.scatter(pdSample, peakPD, marker ='.',color='red', label = "PD signal")
plt.legend(loc='upper left')
plt.xlabel('Sampling')
plt.ylabel('Peak [mV]')
plt.title(dataPath)


plt.figure(figsize=(10, 8), facecolor='white')
# plt.stem(peakNoise, noiseSample, linefmt='blue')
# plt.stem(peakPD, pdSample, linefmt='red')
plt.scatter(noiseSample, riseTimeNoise,  marker ='x', color='blue', label = 'Noise signal')
plt.scatter(pdSample, riseTimePD, marker ='.',color='red', label = "PD signal")
plt.legend(loc='upper left')
plt.xlabel('Sampling')
plt.ylabel('Rise time [us]')
plt.title(dataPath)
plt.show()

a=1