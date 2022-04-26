# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, Conv1D, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.backend import clear_session
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import mean_squared_error, mean_absolute_error
from tensorflow.python.ops.numpy_ops import np_config

from spektral.utils import tic, toc
import numpy as np
from tqdm import tqdm

np_config.enable_numpy_behavior()

# Parameters
n_epoch = 1000  # Number of training epochs
n_patience = 10  # Patience for early stopping
l2_reg = 2.5e-4  # L2 regularization rate
n_trial = 20
dropout = 0.01


def load_data():
    y = np.load("./y.npy")
    cnn_x = np.load("./cnn-embeddings.npy")
    str_x = np.load("./str-embeddings.npy")
    
    random_index = np.random.choice(np.arange(cnn_x.shape[0]), size=1000, replace=False)
    y = y[random_index]
    cnn_x = cnn_x[random_index]
    str_x = str_x[random_index]
    return y, cnn_x, str_x
    
y, cnn_x, str_x = load_data()

str_x = np.expand_dims(str_x, axis=1)

# Train/test
mask_tr = tf.constant([1] * 600 + [0] * 400, dtype=tf.int64)
mask_va = tf.constant([0] * 600 + [1] * 200 + [0] * 200, dtype=tf.int64)
mask_te = tf.constant([0] * 800 + [1] * 200, dtype=tf.int64)


def two_modals():
    # Image branch
    modal_1_x_in = Input(shape=cnn_x.shape[1:])
    modal_1_cov_1 = Conv1D(filters=32,    
                           kernel_size=3,
                           activation="ReLU",
                           kernel_regularizer=l2(l2_reg),
                           bias_regularizer=l2(l2_reg)
                           )(modal_1_x_in)
    modal_1_cov_1 = Dropout(0.1)(modal_1_cov_1)
    modal_1_cov_2 = Conv1D(filters=64,    
                       kernel_size=3,
                       activation="ReLU",
                       kernel_regularizer=l2(l2_reg),
                       bias_regularizer=l2(l2_reg)
                       )(modal_1_cov_1)
    modal_1_cov_2 = Dropout(0.1)(modal_1_cov_2)
    modal_1_cov_3 = Conv1D(filters=128,    
                       kernel_size=3,
                       activation="ReLU",
                       kernel_regularizer=l2(l2_reg),
                       bias_regularizer=l2(l2_reg)
                       )(modal_1_cov_2)
    modal_1_cov_3 = Dropout(0.1)(modal_1_cov_3)
    modal_1_cov_3 = Flatten()(modal_1_cov_3)
    
    # Structure branch
    modal_2_x_in = Input(shape=str_x.shape[1:])
    
    modal_2_cov_1 = Conv1D(filters=32,    
                           kernel_size=1,
                           activation="ReLU",
                           kernel_regularizer=l2(l2_reg),
                           bias_regularizer=l2(l2_reg),
                           )(modal_2_x_in)
    modal_2_cov_1 = Dropout(0.1)(modal_2_cov_1)
    modal_2_cov_2 = Conv1D(filters=64,    
                       kernel_size=1,
                       activation="ReLU",
                       kernel_regularizer=l2(l2_reg),
                       bias_regularizer=l2(l2_reg)
                       )(modal_2_cov_1)
    modal_2_cov_2 = Dropout(0.1)(modal_2_cov_2)
    modal_2_cov_3 = Conv1D(filters=128,    
                       kernel_size=1,
                       activation="ReLU",
                       kernel_regularizer=l2(l2_reg),
                       bias_regularizer=l2(l2_reg)
                       )(modal_2_cov_2)
    modal_2_cov_3 = Dropout(0.1)(modal_2_cov_3)
    modal_2_cov_3 = Flatten()(modal_2_cov_3)
    
    modals_con = Concatenate(axis = 1)([modal_1_cov_3, modal_2_cov_3])
    modals_Dense_1 = Dense(512, activation="ReLU")(modals_con)
    modals_Dense_2 = Dense(32, activation="ReLU")(modals_Dense_1)
    modals_out = Dense(1)(modals_Dense_2)
    
    model = Model(inputs=[modal_1_x_in, modal_2_x_in], outputs=modals_out)
    return model

optimizer = Adam()
loss_fn = MeanSquaredError()


# Trials
te_mse = []
te_mae = []
for trial in tqdm(range(n_trial)):
    clear_session()
    model = two_modals()
    
    y, cnn_x, str_x = load_data()
    str_x = np.expand_dims(str_x, axis=1)
    
    best_val_mse = 99999
    best_test_mse = 0
    best_test_mae = 0
    current_patience = n_patience

    # Training step
    @tf.function
    def train():
        with tf.GradientTape() as tape:
            predictions = model([cnn_x, str_x], training=True)
            loss = loss_fn(y[mask_tr], predictions[mask_tr])
            loss += sum(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    
    
    @tf.function
    def evaluate():
        predictions = model([cnn_x, str_x], training=False)
        losses = []
        mse_lst = []
        mae_lst = []
        for mask in [mask_tr, mask_va, mask_te]:
            loss = loss_fn(y[mask], predictions[mask])
            loss += sum(model.losses)
            losses.append(loss)
            mse = tf.reduce_mean(mean_squared_error(y[mask], predictions[mask]))
            mse_lst.append(mse)
            mae = tf.reduce_mean(mean_absolute_error(y[mask], predictions[mask]))
            mae_lst.append(mae)
        return losses, mse_lst, mae_lst

    tic()
    for epoch in range(1, n_epoch):
        train()
        loss, mse, mae = evaluate()
        if mse[1] < best_val_mse:
            best_val_mse = mse[1]
            best_test_mse = mse[2]
            best_test_mae = mae[2]
            current_patience = n_patience
        else:
            current_patience -= 1
            if current_patience == 0:
                te_mse.append(best_test_mse)
                te_mae.append(best_test_mae)
                break
    toc()
    

# Summary
te_mse_arr = np.array(te_mse)
te_mae_arr = np.array(te_mae)
print("E(mse): " + str(np.mean(te_mse_arr)))
print("STD(mse): " + str(np.std(te_mse_arr)))
print("E(mae): " + str(np.mean(te_mae_arr)))
print("STD(mae): " + str(np.std(te_mae_arr)))