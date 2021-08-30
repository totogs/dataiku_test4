# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
json_prepared = dataiku.Dataset("data_distinct")
df = json_prepared.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd
import time

import os

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Split en dataset de Train et Test

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Split into training, test datasets.
# Since it's timeseries we should do it by date.
test_cutoff_date = df['date'].max() - timedelta(days=7)

test_df = df[df['date'] > test_cutoff_date]
train_df = df[df['date'] <= test_cutoff_date]

#check out the datasets
print('Test dates: {} to {}'.format(test_df['date'].min(), test_df['date'].max()))
print('Train dates: {} to {}'.format(train_df['date'].min(), train_df['date'].max()))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
train_df  = train_df.set_index('date')
test_df = test_df.set_index('date')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
train = train_df
scalers={}
for i in train_df.columns:
    scaler = MinMaxScaler(feature_range=(-1,1))
    s_s = scaler.fit_transform(train[i].values.reshape(-1,1))
    s_s=np.reshape(s_s,len(s_s))
    scalers['scaler_'+ i] = scaler
    train[i]=s_s
test = test_df
for i in test_df.columns:
    scaler = scalers['scaler_'+i]
    s_s = scaler.transform(test[i].values.reshape(-1,1))
    s_s=np.reshape(s_s,len(s_s))
    scalers['scaler_'+i] = scaler
    test[i]=s_s

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Construction des mini-batch

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def split_series(series, n_past, n_future):
  #
  # n_past ==> no of past observations
  #
  # n_future ==> no of future observations
  #
  X, y = list(), list()
  for window_start in range(len(series)):
    past_end = window_start + n_past
    future_end = past_end + n_future
    if future_end > len(series):
      break
    # slicing the past and future parts of the window
    past, future = series[window_start:past_end, :], series[past_end:future_end, :]
    X.append(past)
    y.append(future)
  return np.array(X), np.array(y)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
n_past = 10
n_future = 5
n_features = 10

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
X_train, y_train = split_series(train.values,n_past, n_future)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],n_features))
y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], n_features))
X_test, y_test = split_series(test.values,n_past, n_future)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],n_features))
y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], n_features))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Création du modèle de forecasting

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
encoder = tf.keras.layers.LSTM(100, return_state=True)
encoder_outputs = encoder(encoder_inputs)

encoder_states = encoder_outputs[1:]

#
decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs[0])

#
decoder = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_inputs,initial_state = encoder_states)
decoder_outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(decoder)

#
model = tf.keras.models.Model(encoder_inputs,decoder_outputs)

#
model.summary()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Apprentissage du modèle

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber(), metrics=[tf.keras.metrics.CosineSimilarity(), tf.keras.metrics.MeanAbsoluteError()])
history = model.fit(X_train,y_train,epochs=25,validation_data=(X_test,y_test),batch_size=32,verbose=0,callbacks=[reduce_lr])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print(history.history.keys())
history.history["val_mean_absolute_error"]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
for index,i in enumerate(train_df.columns):
    scaler = scalers['scaler_'+i]
    y_pred[:,:,index]=scaler.inverse_transform(y_pred[:,:,index])
    y_train[:,:,index]=scaler.inverse_transform(y_train[:,:,index])
    y_test[:,:,index]=scaler.inverse_transform(y_test[:,:,index])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Stockage du modèle dans le folder

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
model_json = model.to_json()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
model_folder = dataiku.Folder("dHoUQGRB")
model_folder_info = model_folder.get_info()

model_folder.write_json("/model_json", model_json)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
model_folder.list_paths_in_partition()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# NEW_FOLDER_NAME is the name of the folder you want to create
os.mkdir(model_folder.get_path() + "/" + "okk")