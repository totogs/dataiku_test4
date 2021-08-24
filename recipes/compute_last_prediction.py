# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
json_prepared_filtered = dataiku.Dataset("Json_prepared_filtered")
df = json_prepared_filtered.get_dataframe()

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
del df['Timestamp']
df_cpy  = df.set_index('date')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
val = df_cpy
scalers={}
for i in df_cpy.columns:
    scaler = MinMaxScaler(feature_range=(-1,1))
    s_s = scaler.fit_transform(val[i].values.reshape(-1,1))
    s_s=np.reshape(s_s,len(s_s))
    scalers['scaler_'+ i] = scaler
    val[i]=s_s

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
val_to_pred = val.values
val_to_pred = val_to_pred.reshape((1, val_to_pred.shape[0],val_to_pred.shape[1]))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
handle = dataiku.Folder("models")
json_config = handle.read_json("model_json")
model = keras.models.model_from_json(json_config)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
val_pred = model.predict(val_to_pred)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
val = val_pred.squeeze()
val.shape

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
lastdate = df["date"].max()
print(lastdate)
for i in range(len(val)):
    lastdate = lastdate + timedelta(minutes=15)
    df.append()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

last_prediction_df = json_prepared_filtered_df # For this sample code, simply copy input to output


# Write recipe outputs
last_prediction = dataiku.Dataset("last_prediction")
last_prediction.write_with_schema(last_prediction_df)