# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.backend import clear_session
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import mean_squared_error, mean_absolute_error
from tensorflow.python.ops.numpy_ops import np_config

from spektral.layers import GATConv
from spektral.data import Graph, Dataset
from spektral.utils import tic, toc

import numpy as np
from tqdm import tqdm

np_config.enable_numpy_behavior()

# Parameters
channels_1 = 64  # Number of channels in each head of the first GAT layer
n_attn_heads = 128  # Number of attention heads in first GAT layer
n_epoch = 1000  # Number of training epochs
n_patience = 10  # Patience for early stopping
dropout = 0.01  # Dropout rate for the features and adjacency matrix
l2_reg = 2.5e-4  # L2 regularization rate
n_trial = 20


class Structure(Dataset):       
    def __init__(self, nodes, feats, **kwargs):
        self.nodes = nodes
        self.feats = feats

        super().__init__(**kwargs)
    
    def read(self):
        y = np.load("./y.npy")
        str_x = np.load("./str-embeddings.npy")
        str_a = np.load("./str-matrix.npy")
        
        random_index = np.random.choice(np.arange(str_x.shape[0]), size=1000, replace=False)
        str_x = str_x[random_index]
        str_a = str_a[random_index]
        str_a = str_a[:, random_index]
        y = y[random_index]
        
        return [Graph(x=str_x, a=str_a, y=y)]

str_data = Structure(1000, 665)
str_graph = str_data[0]

# Train/test
mask_tr = tf.constant([1] * 600 + [0] * 400, dtype=tf.int64)
mask_va = tf.constant([0] * 600 + [1] * 200 + [0] * 200, dtype=tf.int64)
mask_te = tf.constant([0] * 800 + [1] * 200, dtype=tf.int64)


def str_unimodal():
    modal_2_x_in = Input(shape=(str_data.n_node_features,))
    modal_2_a_in = Input(shape=(str_data.n_nodes,))
    
    modal_2_gat_1 = GATConv(
            channels_1,
            attn_heads=n_attn_heads,
            concat_heads=False,
            dropout_rate=dropout,
            activation="sigmoid",
            kernel_regularizer=l2(l2_reg),
            attn_kernel_regularizer=l2(l2_reg),
            bias_regularizer=l2(l2_reg),        
        )([modal_2_x_in, modal_2_a_in])
    modal_2_Dense_1 = Dense(32, activation="ReLU")(modal_2_gat_1)
    modal_2_out = Dense(1)(modal_2_Dense_1)
    
    model = Model(inputs=[modal_2_x_in, modal_2_a_in], outputs=modal_2_out)
    return model

optimizer = Adam()
loss_fn = MeanSquaredError()

# Trials
te_mse = []
te_mae = []
for trial in tqdm(range(n_trial)):
    clear_session()
    model = str_unimodal()
    
    str_data = Structure(1000, 665)
    str_graph = str_data[0]
    
    best_val_mse = 99999
    best_test_mse = 0
    best_test_mae = 0
    current_patience = n_patience

    # Training step
    @tf.function
    def train():
        with tf.GradientTape() as tape:
            predictions = model([str_graph.x, str_graph.a], training=True)
            loss = loss_fn(str_graph.y[mask_tr], predictions[mask_tr])
            loss += sum(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    
    
    @tf.function
    def evaluate():
        predictions = model([str_graph.x, str_graph.a], training=False)
        losses = []
        mse_lst = []
        mae_lst = []
        for mask in [mask_tr, mask_va, mask_te]:
            loss = loss_fn(str_graph.y[mask], predictions[mask])
            loss += sum(model.losses)
            losses.append(loss)
            mse = tf.reduce_mean(mean_squared_error(str_graph.y[mask], predictions[mask]))
            mse_lst.append(mse)
            mae = tf.reduce_mean(mean_absolute_error(str_graph.y[mask], predictions[mask]))
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