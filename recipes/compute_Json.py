# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import json

# Read recipe inputs
disk_usage = dataiku.Dataset("disk_usage")
df = disk_usage.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df.info()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
for key, value in df["data"].items():
    print(value)
    dict = json.loads(value)
dict

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df = pd.json_normalize(dict["result"])
df = df.explode('values')
df[['Timestamp','CPU_Usage']] = pd.DataFrame(df['values'].tolist(),index=df.index)
df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df.rename(columns={'metric.hostname': 'Broker', 'metric.dc': 'Data_Center','metric.mountpoint':'Disk'}, inplace=True)
df = df.drop(['metric.cluster', 'metric.job','metric.device','metric.fstype','values'], axis=1)
df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
json = dataiku.Dataset("Json")
json.write_with_schema(df)