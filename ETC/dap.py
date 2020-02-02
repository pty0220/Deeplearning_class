import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import load_model
from matplotlib import colors
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


seed_number = 5
model_path = "../model/TSNE"+str(seed_number)+".csv"

TSNE_data = pd.read_csv(model_path)
TSNE_data = TSNE_data.iloc[:, 1:].values
idx_unknown = np.min(np.where(TSNE_data[:,-1]==2))

X = TSNE_data[:idx_unknown, :-1]
y = TSNE_data[:idx_unknown, -1]

unknown_X = TSNE_data[idx_unknown:, :-1]
unknown_y = TSNE_data[idx_unknown:, -1]

# #############################################################################
# Plot functions
cmap = colors.LinearSegmentedColormap(
    'red_blue_classes',
    {'red': [(0, 1, 1), (1, 0.7, 0.7)],
     'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
     'blue': [(0, 0.7, 0.7), (1, 1, 1)]})

plt.cm.register_cmap(cmap=cmap)



def plot_data(lda, X, y, y_pred):

    tp = (y == y_pred)  # True Positive
    tp0, tp1 = tp[y == 0], tp[y == 1]
    X0, X1 = X[y == 0], X[y == 1]
    X0_tp, X0_fp = X0[tp0], X0[~tp0]
    X1_tp, X1_fp = X1[tp1], X1[~tp1]

    # class 0: dots
    plt.scatter(X0_tp[:, 0], X0_tp[:, 1], marker='.', color='red')
    plt.scatter(X0_fp[:, 0], X0_fp[:, 1], marker='x',
                s=20, color='#990000')  # dark red

    # class 1: dots
    plt.scatter(X1_tp[:, 0], X1_tp[:, 1], marker='.', color='blue')
    plt.scatter(X1_fp[:, 0], X1_fp[:, 1], marker='x',
                s=20, color='#000099')  # dark blue

    # class 0 and 1 : areas
    nx, ny = 100, 100
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
    Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes',
                   norm=colors.Normalize(0., 1.), zorder=0)
    plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='white')

def plot_data2(lda, X, y, y_pred, unknown_X, unknown_y, unknown_NN_y_pred):

    tp = (y == y_pred)  # True Positive
    tp0, tp1 = tp[y == 0], tp[y == 1]
    X0, X1 = X[y == 0], X[y == 1]
    X0_tp, X0_fp = X0[tp0], X0[~tp0]
    X1_tp, X1_fp = X1[tp1], X1[~tp1]

    un_y0 = unknown_NN_y_pred[unknown_NN_y_pred == 0]
    un_y1 = unknown_NN_y_pred[unknown_NN_y_pred == 1]

    unX0 = unknown_X[unknown_NN_y_pred == 0]
    unX1 = unknown_X[unknown_NN_y_pred == 1]

    # class 0: dots
    plt.scatter(X0_tp[:, 0], X0_tp[:, 1], marker='.', color='red')
    plt.scatter(X0_fp[:, 0], X0_fp[:, 1], marker='x',
                s=20, color='#990000')  # dark red

    # class 1: dots
    plt.scatter(X1_tp[:, 0], X1_tp[:, 1], marker='.', color='blue')
    plt.scatter(X1_fp[:, 0], X1_fp[:, 1], marker='x',
                s=20, color='#000099')  # dark blue


    plt.scatter(unX0[:, 0], unX0[:, 1], marker='o', color='#999990')

    plt.scatter(unX1[:, 0], unX1[:, 1], marker='+', color='#000009')


    # class 0 and 1 : areas
    nx, ny = 100, 100
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
    Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes',
                   norm=colors.Normalize(0., 1.), zorder=0)
    plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='white')





### QDA prediction
qda = QuadraticDiscriminantAnalysis(store_covariance=False)
qda_y_pred = qda.fit(X, y).predict(X)
plt.figure(figsize=(10, 8), facecolor='white')
plot_data(qda, X, y, qda_y_pred)

### NN prediction
model = load_model('../model/TSNE2NN.h5')
NN_y_pred = model.predict(X)
NN_y_pred = np.round(NN_y_pred[:, 1])

unknown_NN_y_pred = model.predict(unknown_X)
unknown_NN_y_pred = np.round(unknown_NN_y_pred[:, 1])


data = pd.read_csv('../pre_data/PD_noise_Unknown.csv')
num_data = data.iloc[:, :].values
num_data[idx_unknown:,-1] = unknown_NN_y_pred

DATA = pd.DataFrame(data = num_data)
DATA.to_csv("../pre_data/whole_data_unknown_label.csv", index = False, header = False)




plt.figure(figsize=(10, 8), facecolor='white')
plot_data(model, X, y, NN_y_pred)

plt.figure(figsize=(10, 8), facecolor='white')
plot_data2(model, X, y, NN_y_pred, unknown_X, unknown_y, unknown_NN_y_pred)


plt.show()