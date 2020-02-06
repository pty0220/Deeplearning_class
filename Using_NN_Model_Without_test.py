import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from keras.models import load_model

def result_show(idx):

    if idx == 1:
        result_text = 'Noise'
    else:
        result_text = 'PD'

    return result_text


# Import and split data
data = pd.read_csv('../pre_data/dangin_CH3_whole_data.csv')
time = pd.read_csv('../pre_data/Times.csv')

"""Model Building"""
# Split features and labels
x = data.iloc[:, :].values
y = data.iloc[:, -1].values

time = time.iloc[:,:].values
time = np.squeeze(time)

# Build Neural Network

model = load_model('../model/final_NN_model.h5')
y_hat = model.predict(x)
y_hat = np.round(y_hat[:, 1])
#print(metrics.accuracy_score(y_hat,y))

num_plt = 5
num_plt = num_plt+1
#rand_idx = [100, 100, 76, 112, 21, 24]
rand_idx = np.random.randint(0, 200, size=num_plt)

fig = plt.figure(figsize=(5,10))


for i in range(1,num_plt):

    ax = fig.add_subplot(num_plt, 1, i)
    ax.plot(time, x[rand_idx[i],:-1])
    one_x = x[rand_idx[i], :]

    y_hat = model.predict(np.array([one_x,]))
    y_hat = np.round(y_hat[:, 1])

    predict_result = result_show(y_hat)

    plt.xlabel('time')
    plt.ylabel('voltage (mV)')
    #plt.xlim((0,3000))
    #plt.ylim((-200,200))
    plt.title('NN Prediction: '+predict_result)
    ax.set_aspect('auto')
    plt.tight_layout()  # not strictly part of the question

#plt.tight_layout()
plt.show()



a=1