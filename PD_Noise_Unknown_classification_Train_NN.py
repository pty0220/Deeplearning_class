# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics


from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from keras.layers import Dropout
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import History
from keras.models import load_model



# Import and split data
data = pd.read_csv('../pre_data/PD_Noise_Unknown.csv')



"""Model Building"""
# Split features and labels
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)

# Build Neural Network

n_cols = x_train.shape[1]
y_train = to_categorical(y_train, 3)
y_test = to_categorical(y_test, 3)



model = Sequential()

model.add(Dense(500, activation='relu', input_dim=n_cols))
#model.add(Dropout(0.5))

model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(500, activation='relu'))
#model.add(Dropout(0.5))

model.add(Dense(400, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(200, activation='relu'))
#model.add(Dropout(0.5))

model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(50, activation='relu'))

model.add(Dense(25, activation='relu'))

model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test,y_test))

y_hat = model.predict(x_test)
y_hat = np.round(y_hat[:, 1])


plt.plot(hist.history['loss'], 'y', label='train loss')
plt.plot(hist.history['val_loss'], 'r', label='val loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper left')
plt.show()



plt.plot(hist.history['acc'], 'b', label='train acc')
plt.plot(hist.history['val_acc'], 'g', label='val acc')
plt.xlabel('epoch')
plt.ylabel('accuray')
plt.legend(loc='lower left')
plt.show()

print(" \n")
print(" \n")

model.save('../model/NN_model_unknown.h5')
a=1