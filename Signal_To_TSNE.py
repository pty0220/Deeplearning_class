import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

from  sklearn.manifold import TSNE


## singal -> TSNE -> sklearn 을 이용한 TSNE data 의 classifiaction


data = pd.read_csv('../pre_data/PD_Noise_Unknown.csv')
seed_number = 3

np.random.seed(seed_number)
# Split features and labels
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

sample_size = x.shape[0]

model_path = "../model/TSNE"+str(seed_number)+".csv"
check = os.path.isfile(model_path)
print(check)

random = np.random.rand(sample_size, 2)

if not os.path.isfile(model_path):

    print("make model")
    # TSNE 해석 파트
    model = TSNE(learning_rate=100, init=random)
    TSNE_data = model.fit_transform(x)

    temp = np.zeros((TSNE_data.shape[0],3))
    temp[:,:-1] = TSNE_data
    temp[:,-1] = y
    TSNE_data = temp

    DATA = pd.DataFrame(data = TSNE_data)
    DATA.to_csv(model_path)

else:
    print("read model")
    TSNE_data = pd.read_csv(model_path)
    TSNE_data = TSNE_data.iloc[:,1:].values


xs = TSNE_data[:,0]
ys = TSNE_data[:,1]

plt.figure(figsize=(10, 8), facecolor='white')

for i in range(xs.shape[0]):
    #print(i)
    if y[i] ==0:
        plt.text(xs[i], ys[i], str(i), color=[1,0,0], fontdict={'size': 7})
    if y[i] ==1:
        plt.text(xs[i], ys[i], str(i), color=[0,1,0], fontdict={ 'size': 7})
    if y[i] ==2:
        plt.text(xs[i], ys[i], str(i), color= [0,0,1], fontdict={ 'size': 7})

plt.xlim(np.min(xs)-10,np.max(xs)+10)
plt.ylim(np.min(ys)-10,np.max(ys)+10)

plt.show()

a=1