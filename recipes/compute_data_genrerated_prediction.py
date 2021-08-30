# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
data_generated = dataiku.Dataset("data_generated")
data_generated_df = data_generated.get_dataframe()

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
df_cpy  = data_generated_df.set_index('date')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
data = df_cpy
scalers={}
for i in df_cpy.columns:
    scaler = MinMaxScaler(feature_range=(-1,1))
    s_s = scaler.fit_transform(data[i].values.reshape(-1,1))
    s_s=np.reshape(s_s,len(s_s))
    scalers['scaler_'+ i] = scaler
    data[i]=s_s

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
data_to_pred = data.values
data_to_pred = data_to_pred.reshape((1, data_to_pred.shape[0],data_to_pred.shape[1]))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
handle = dataiku.Folder("model_forecast")
json_config = handle.read_json("actual/model_json")
model = keras.models.model_from_json(json_config)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
data_pred = model.predict(data_to_pred)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
for index,i in enumerate(df_cpy.columns):
    scaler = scalers['scaler_'+i]
    data_pred[:,:,index]=scaler.inverse_transform(data_pred[:,:,index])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
data_pred = data_pred.squeeze()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_pred = pd.DataFrame(data_pred, columns=df_cpy.columns)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
lastdate = data_generated_df["date"].max()
listdate = []
for i in range(len(df_pred)):
    lastdate = lastdate + timedelta(minutes=15)
    listdate.append(lastdate)
    
df_pred["date"] = listdate

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
data_genrerated_prediction = dataiku.Dataset("data_genrerated_prediction")
data_genrerated_prediction.write_with_schema(df_pred)