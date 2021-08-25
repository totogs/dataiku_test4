# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
json_prepared = dataiku.Dataset("json_stacked_distinct")
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

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df.head()


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# Split into training, validation and test datasets.
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
train_df.head()

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
n_features = 30

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
X_train, y_train = split_series(train.values,n_past, n_future)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],n_features))
y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], n_features))
X_test, y_test = split_series(test.values,n_past, n_future)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],n_features))
y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], n_features))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
encoder_l1 = tf.keras.layers.LSTM(100, return_state=True)
encoder_outputs1 = encoder_l1(encoder_inputs)

encoder_states1 = encoder_outputs1[1:]

#
decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs1[0])

#
decoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_inputs,initial_state = encoder_states1)
decoder_outputs1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(decoder_l1)

#
model_e1d1 = tf.keras.models.Model(encoder_inputs,decoder_outputs1)

#
model_e1d1.summary()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)
model_e1d1.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber())
history_e1d1=model_e1d1.fit(X_train,y_train,epochs=25,validation_data=(X_test,y_test),batch_size=32,verbose=0,callbacks=[reduce_lr])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
pred_e1d1=model_e1d1.predict(X_test)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
for index,i in enumerate(train_df.columns):
    scaler = scalers['scaler_'+i]
    pred_e1d1[:,:,index]=scaler.inverse_transform(pred_e1d1[:,:,index])
    y_train[:,:,index]=scaler.inverse_transform(y_train[:,:,index])
    y_test[:,:,index]=scaler.inverse_transform(y_test[:,:,index])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
pred_e1d1

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
X_test.shape

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
y_test.shape

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
pred_e1d1.shape


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
json_config = model_e1d1.to_json()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE


# Write recipe outputs
model_forecast = dataiku.Folder("dHoUQGRB")
model_forecast_info = model_forecast.get_info()

model_forecast.write_json("model_json", json_config)
