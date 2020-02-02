import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from scipy.signal import find_peaks
from scipy.misc import electrocardiogram
from sklearn.preprocessing import normalize


# Import and split data
data = pd.read_csv('../pre_data/PD_Noise_Unknown_labeled.csv')
time = pd.read_csv('../pre_data/Times.csv')

"""Model Building"""
# Split features and labels
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

peak = x.max(axis=1)
riseTime = time[x.argmax(axis=1)]



a=1