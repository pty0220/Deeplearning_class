import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from scipy.signal import find_peaks
from scipy.misc import electrocardiogram
from sklearn.preprocessing import normalize

def findRisePoint(signal, cutValue):

    absSignal = np.abs(signal)
    cut = absSignal[absSignal>cutValue]
    riseValue = cut[0]
    index = np.where(absSignal == riseValue)[0][0]-1
    return index

# Import and split data
data = pd.read_csv('../pre_data/PD_Noise_Unknown_labeled.csv')
time = pd.read_csv('../pre_data/Times.csv')

"""Model Building"""
# Split features and labels
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

maximum = x.max(axis=1)
idx = x.argmax(axis=1)
normalX = 100*x/maximum[:,None]

oneX = normalX[700, :-1]
oneX = np.squeeze(oneX)

risepoint = findRisePoint(oneX, 2)
peaks, _ = find_peaks(oneX, height=10)
plt.plot(oneX)
plt.plot(peaks, oneX[peaks], "x")
plt.plot(risepoint, oneX[risepoint], "o")
plt.plot(np.zeros_like(oneX), "--", color="gray")
plt.show()

