# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.backend import clear_session
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.metrics import mean_absolute_error
from tensorflow.python.ops.numpy_ops import np_config

from spektral.layers import GATConv
from spektral.data import Graph, Dataset
from spektral.utils import tic, toc

import numpy as np
from tqdm import tqdm


np_config.enable_numpy_behavior()
 

# Parameters
channels_1 = 16  # Number of channels in each head of the first GAT layer
n_attn_heads = 16  # Number of attention heads in first GAT layer
n_epoch = 1000  # Number of training epochs
n_patience = 50  # Patience for early stopping
dropout = 0.1  # Dropout rate for the features and adjacency matrix
l2_reg = 0.001  # L2 regularization rate
n_trial = 5
seeds = 42 


class Video(Dataset):       
    def __init__(self, nodes, feats, **kwargs):
        self.nodes = nodes
        self.feats = feats

        super().__init__(**kwargs)
    
    def read(self):
        y = np.load("./y.npy")
        video_x = np.load("./video-embeddings.npy")
        video_x = video_x.reshape((video_x.shape[0], -1))
        video_a = np.load("./video-matrix.npy")  
        
        index = np.arange(video_x.shape[0])
        np.random.seed(seeds)
        np.random.shuffle(index)
        video_x = video_x[index, :]
        video_a = video_a[index, :]
        y = y[index, :]
        return [Graph(x=video_x, a=video_a, y=y)]
    
    
class Audio(Dataset):       
    def __init__(self, nodes, feats, **kwargs):
        self.nodes = nodes
        self.feats = feats

        super().__init__(**kwargs)
    
    def read(self):
        y = np.load("./y.npy")
        audio_x = np.load("./audio-embeddings.npy")
        audio_x = audio_x.reshape((audio_x.shape[0], -1))
        audio_a = np.load("./audio-matrix.npy")
        
        index = np.arange(audio_x.shape[0])
        np.random.seed(seeds)
        np.random.shuffle(index)
        audio_x = audio_x[index, :]
        audio_a = audio_a[index, :]
        y = y[index, :]
        return [Graph(x=audio_x, a=audio_a, y=y)]

video_data = Video(61, 202752)
audio_data = Audio(61, 55140)

video_graph = video_data[0]
audio_graph = audio_data[0]

# Train/test
mask_tr = tf.constant([1] * 37 * 9 + [0] * 24 * 9, dtype=tf.int32).reshape(-1, 9)
mask_va = tf.constant([0] * 37 * 9 + [1] * 12 * 9 + [0] * 12 * 9, dtype=tf.int32).reshape(-1, 9)
mask_te = tf.constant([0] * 49 * 9 + [1] * 12 * 9, dtype=tf.int32).reshape(-1, 9)


def two_modals():
    # Video branch
    modal_1_x_in = Input(shape=(video_data.n_node_features,))
    modal_1_a_in = Input(shape=(video_data.n_nodes,))
    
    modal_1_gat_1 = GATConv(
            channels_1,
            attn_heads=n_attn_heads,
            concat_heads=False,
            dropout_rate=dropout,
            activation="sigmoid",
            kernel_regularizer=l2(l2_reg),
            attn_kernel_regularizer=l2(l2_reg),
            bias_regularizer=l2(l2_reg),        
        )([modal_1_x_in, modal_1_a_in])
    modal_1_Dense_1 = Dense(8, activation="ReLU")(modal_1_gat_1)
    
    # Audio branch
    modal_2_x_in = Input(shape=(audio_data.n_node_features,))
    modal_2_a_in = Input(shape=(audio_data.n_nodes,))
    
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
    modal_2_Dense_1 = Dense(8, activation="ReLU")(modal_2_gat_1)
    
    modals_con = Concatenate(axis = 1)([modal_1_Dense_1, modal_2_Dense_1])
    modals_out = Dense(9, activation="sigmoid")(modals_con)
    
    model = Model(inputs=[modal_1_x_in, modal_1_a_in, modal_2_x_in, modal_2_a_in], outputs=modals_out)
    return model

optimizer = RMSprop()
loss_fn = MeanAbsoluteError()

def five_folds(i):
    if i == 0:
        mask_tr = tf.constant([1] * 37 * 9 + [0] * 24 * 9, dtype=tf.int32).reshape(-1, 9)
        mask_va = tf.constant([0] * 37 * 9 + [1] * 12 * 9 + [0] * 12 * 9, dtype=tf.int32).reshape(-1, 9)
        mask_te = tf.constant([0] * 49 * 9 + [1] * 12 * 9, dtype=tf.int32).reshape(-1, 9)
    elif i == 1:
        mask_tr = tf.constant([1] * 37 * 9 + [0] * 24 * 9, dtype=tf.int32).reshape(-1, 9)
        mask_va = tf.constant([0] * 49 * 9 + [1] * 12 * 9, dtype=tf.int32).reshape(-1, 9)
        mask_te = tf.constant([0] * 37 * 9 + [1] * 12 * 9 + [0] * 12 * 9, dtype=tf.int32).reshape(-1, 9)
    elif i == 2:
        mask_tr = tf.constant([1] * 25 * 9 + [0] * 12 * 9 + [1] * 12 * 9 + [0] * 12 * 9, dtype=tf.int32).reshape(-1, 9)
        mask_va = tf.constant([0] * 49 * 9 + [1] * 12 * 9, dtype=tf.int32).reshape(-1, 9)
        mask_te = tf.constant([0] * 25 * 9 + [1] * 12 * 9 + [0] * 24 * 9, dtype=tf.int32).reshape(-1, 9)
    elif i == 3:
        mask_tr = tf.constant([1] * 13 * 9 + [0] * 12 * 9 + [1] * 24 * 9 + [0] * 12 * 9, dtype=tf.int32).reshape(-1, 9)
        mask_va = tf.constant([0] * 49 * 9 + [1] * 12 * 9, dtype=tf.int32).reshape(-1, 9)
        mask_te = tf.constant([0] * 13 * 9 + [1] * 12 * 9 + [0] * 36 * 9, dtype=tf.int32).reshape(-1, 9)
    elif i == 4:
        mask_tr = tf.constant([1] * 1 * 9 + [0] * 12 * 9 + [1] * 36 * 9 + [0] * 12 * 9, dtype=tf.int32).reshape(-1, 9)
        mask_va = tf.constant([0] * 49 * 9 + [1] * 12 * 9, dtype=tf.int32).reshape(-1, 9)
        mask_te = tf.constant([0] * 1 * 9 + [1] * 12 * 9 + [0] * 48 * 9, dtype=tf.int32).reshape(-1, 9)        
    else:
        print("Wrong index!")
    return mask_tr, mask_va, mask_te

# Trials
te_zero = []
te_mae = []

for trial in tqdm(range(n_trial)):
    clear_session()
    model = two_modals()
    
    best_val_mae = 99999
    best_test_zero_one = 0
    best_test_mae = 0
    current_patience = n_patience
    
    mask_tr, mask_va, mask_te = five_folds(trial)

    # Training step
    @tf.function
    def train():
        with tf.GradientTape() as tape:
            predictions = model([video_graph.x, video_graph.a, audio_graph.x, audio_graph.a], training=True)
            predictions_tr = tf.math.multiply(predictions, tf.cast(mask_tr, tf.float32))
            y_tr = tf.math.multiply(video_graph.y, mask_tr)
            loss = loss_fn(y_tr, predictions_tr)
            loss += sum(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    
    
    @tf.function
    def evaluate(): 
        predictions = model([video_graph.x, video_graph.a, audio_graph.x, audio_graph.a], training=False)
        losses = []
        zero_one_lst = []
        mae_lst = []
        for mask in [mask_tr, mask_va, mask_te]:
            predictions_ma = tf.math.multiply(predictions, tf.cast(mask, tf.float32))
            y_ma = tf.math.multiply(video_graph.y, mask)
            loss = loss_fn(y_ma, predictions_ma)
            loss += sum(model.losses)
            losses.append(loss)
            temp = tf.math.floor(tf.reduce_mean(tf.cast(tf.equal(y_ma, tf.cast(tf.math.rint(predictions_ma), tf.int32)), tf.float32), 1))
            zero_one = tf.math.reduce_sum(1 - temp) / temp.shape[0]
            zero_one_lst.append(zero_one)
            mae = tf.reduce_mean(mean_absolute_error(video_graph.y[mask] - predictions[mask]))
            mae_lst.append(mae)
        return losses, zero_one_lst, mae_lst


    tic()
    for epoch in range(1, n_epoch):
        train()
        loss, zero_one, mae = evaluate()
        if mae[1] < best_val_mae:
            best_val_mae = mae[1]
            best_test_zero_one = zero_one[2]
            best_test_mae = mae[2]
            current_patience = n_patience
        else:
            current_patience -= 1
            if current_patience == 0:
                te_zero.append(best_test_zero_one)
                te_mae.append(best_test_mae)
                break
    toc()
    

# Summary
te_zero_arr = np.array(te_zero)
te_mae_arr = np.array(te_mae)
print("E(zero): " + str(np.mean(te_zero_arr)))
print("STD(zero): " + str(np.std(te_zero_arr)))
print("E(mae): " + str(np.mean(te_mae_arr)))
print("STD(mae): " + str(np.std(te_mae_arr)))