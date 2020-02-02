import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from keras.layers import Dense
from keras.models import Sequential


## singal -> TSNE -> NN 을 이용한 TSNE data 의 classifiaction

seed_number = 5
model_path = "../model/TSNE"+str(seed_number)+".csv"

TSNE_data = pd.read_csv(model_path)
TSNE_data = TSNE_data.iloc[:, 1:].values
idx_unknown = np.min(np.where(TSNE_data[:,-1]==2))

x = TSNE_data[:idx_unknown, :-1]
y = TSNE_data[:idx_unknown, -1]

n_cols = x.shape[1]
y_cate = to_categorical(y, 2)

model = Sequential()
model.add(Dense(10, activation='relu', input_dim=n_cols))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
hist = model.fit(x, y_cate, epochs=40, batch_size=32, validation_data=(x, y_cate))

y_hat = model.predict(x)
y_hat = np.round(y_hat[:, 1])

plt.plot(hist.history['loss'], 'y', label='train loss')
plt.plot(hist.history['val_loss'], 'r', label='val loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper left')
plt.show()

plt.plot(hist.history['accuracy'], 'b', label='train acc')
plt.plot(hist.history['val_accuracy'], 'g', label='val acc')
plt.xlabel('epoch')
plt.ylabel('accuray')
plt.legend(loc='lower left')
plt.show()



model.save('../model/TSNE2NN.h5')
