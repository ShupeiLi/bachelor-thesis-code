# -*- coding: utf-8 -*-

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

# Parameters
n_trial = 20


def load_data():
    y = np.load("/y.npy")
    cnn_x = np.load("./cnn-embeddings.npy")
    cnn_x = cnn_x.reshape((cnn_x.shape[0], -1))
    str_x = np.load("./str-embeddings.npy")
    
    random_index = np.random.choice(np.arange(cnn_x.shape[0]), size=1000, replace=False)
    y = y[random_index]
    cnn_x = cnn_x[random_index]
    str_x = str_x[random_index]
    return y, cnn_x, str_x
    
y, cnn_x, str_x = load_data()
x = np.concatenate((cnn_x, str_x), axis=1)

def xgb_reg():
    model = xgb.XGBRFRegressor()
    model.fit(x[:600, :], y[:600])
    y_pre = model.predict(x[800:, :])
    mse = mean_squared_error(y[800:], y_pre)
    mae = mean_absolute_error(y[800:], y_pre)
    return mse, mae
    
# Trials
te_mse = []
te_mae = []
for trial in tqdm(range(n_trial)):
    y, cnn_x, str_x = load_data()
    x = np.concatenate((cnn_x, str_x), axis=1)
    mse, mae = xgb_reg()
    te_mse.append(mse)
    te_mae.append(mae)

# Summary
te_mse_arr = np.array(te_mse)
te_mae_arr = np.array(te_mae)
print("E(mse): " + str(np.mean(te_mse_arr)))
print("STD(mse): " + str(np.std(te_mse_arr)))
print("E(mae): " + str(np.mean(te_mae_arr)))
print("STD(mae): " + str(np.std(te_mae_arr)))