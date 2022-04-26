# -*- coding: utf-8 -*-

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import zero_one_loss

seeds = 42


y = np.load("./y.npy")
video_x = np.load("./video-embeddings.npy")
audio_x = np.load("./audio-embeddings.npy")
video_x = video_x.reshape((video_x.shape[0], -1))
audio_x = audio_x.reshape((audio_x.shape[0], -1))
index = np.arange(audio_x.shape[0])
np.random.seed(seeds)
np.random.shuffle(index)
video_x = video_x[index, :]
audio_x = audio_x[index, :]
y = y[index, :]
x = np.concatenate((video_x, audio_x), axis=1)


def five_folds(i):
    mask_arr = np.arange(61)
    if i == 0:
        mask_tr = mask_arr[:49]
        mask_te = mask_arr[49:]    
    if i == 1:
        mask_tr = np.concatenate((mask_arr[:37], mask_arr[49:]))
        mask_te = mask_arr[37:49]
    if i == 2:
        mask_tr = np.concatenate((mask_arr[:25], mask_arr[37:]))
        mask_te = mask_arr[25:37]
    if i == 3:
        mask_tr = np.concatenate((mask_arr[:13], mask_arr[25:]))
        mask_te = mask_arr[13:25]
    if i == 4:
        mask_tr = np.concatenate((mask_arr[:1], mask_arr[13:]))
        mask_te = mask_arr[1:13]
    return mask_tr, mask_te

zero_lst = []
mae_lst = []
for i in range(5):
    mask_tr, mask_te = five_folds(i)
    neigh = KNeighborsClassifier(10)
    neigh.fit(x[mask_tr], y[mask_tr])
    y_pred = neigh.predict(x[mask_te])
    zero_lst.append(zero_one_loss(y[mask_te], y_pred))
    mae_lst.append(np.mean(np.sum(np.abs(y[mask_te] - y_pred), axis=1)))
    
zero_arr = np.array(zero_lst)
mae_arr = np.array(mae_lst)
print("E(zero): " + str(np.mean(zero_arr)))
print("STD(zero): " + str(np.std(zero_arr)))
print("E(mae): " + str(np.mean(mae_arr)))
print("STD(mae): " + str(np.std(mae_arr)))