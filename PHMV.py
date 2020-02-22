import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from keras.models import load_model




phmv = pd.read_csv('../pre_data/CH2_PHMV.csv')

# Split features and labels
ph = phmv.iloc[:, 0].values
mv = phmv.iloc[:, 1].values

plt.scatter(ph, mv, marker='.', color='black')
plt.xlabel('Ph')
plt.ylabel('mV')
plt.show()


data = pd.read_csv('../pre_data/CH1_whole_data.csv')
inputSignal = data.iloc[:, :].values


model = load_model('../model/final_NN_model.h5')
y_hat = model.predict(inputSignal)
y_hat = np.round(y_hat[:, 1])

phmvLabel = np.zeros((y_hat.shape[0],3))
phmvLabel[:,0] = y_hat
phmvLabel[:,1:] = phmv

a=1
