# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
disk_usage_25_08 = dataiku.Dataset("disk_usage_25_08")
df = disk_usage_25_08.get_dataframe()


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
df['Broker_Disk'] = df['Broker'] + df['Disk']
df = df.drop(['metric.cluster', 'metric.job','metric.device','metric.fstype','values','Broker','Disk'], axis=1)
df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
list(df['Broker_Disk'].unique())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
dfx = pd.pivot_table ( df, index=['Timestamp'], columns = df.groupby(['Timestamp']).cumcount().add(1), values = ['CPU_Usage'], aggfunc = 'sum')
dfx.columns = list(df['Broker_Disk'].unique())
dfx = dfx.reset_index()
dfx

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df = dfx



# Write recipe outputs
json_2 = dataiku.Dataset("json_2")
json_2.write_with_schema(df)
