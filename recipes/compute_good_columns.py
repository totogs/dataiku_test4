# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import json

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read recipe inputs
prepared = dataiku.Dataset("prepared")
df = prepared.get_dataframe()
df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#On fait des sous daframes pour pouvoir faire les modifications sur tout les brokers en memes temps

df0 = df[["data.result.0_metric","data.result.0_values"]]
df1 = df[["data.result.1_metric","data.result.1_values"]]
df2 = df[["data.result.2_metric","data.result.2_values"]]
df3 = df[["data.result.3_metric","data.result.3_values"]]
df4 = df[["data.result.4_metric","data.result.4_values"]]

df5 = df[["data.result.5_metric","data.result.5_values"]]
df6 = df[["data.result.6_metric","data.result.6_values"]]
df7 = df[["data.result.7_metric","data.result.7_values"]]
df8 = df[["data.result.8_metric","data.result.8_values"]]
df9 = df[["data.result.9_metric","data.result.9_values"]]

df10 = df[["data.result.10_metric","data.result.10_values"]]
df11 = df[["data.result.11_metric","data.result.11_values"]]
df12 = df[["data.result.12_metric","data.result.12_values"]]
df13 = df[["data.result.13_metric","data.result.13_values"]]
df14 = df[["data.result.14_metric","data.result.14_values"]]

df15 = df[["data.result.15_metric","data.result.15_values"]]
df16 = df[["data.result.16_metric","data.result.16_values"]]
df17 = df[["data.result.17_metric","data.result.17_values"]]
df18 = df[["data.result.18_metric","data.result.18_values"]]
df19 = df[["data.result.19_metric","data.result.19_values"]]

df20 = df[["data.result.20_metric","data.result.20_values"]]
df21 = df[["data.result.21_metric","data.result.21_values"]]
df22 = df[["data.result.22_metric","data.result.22_values"]]
df23 = df[["data.result.23_metric","data.result.23_values"]]
df24 = df[["data.result.24_metric","data.result.24_values"]]

df25 = df[["data.result.25_metric","data.result.25_values"]]
df26 = df[["data.result.26_metric","data.result.26_values"]]
df27 = df[["data.result.27_metric","data.result.27_values"]]
df28 = df[["data.result.28_metric","data.result.28_values"]]
df29 = df[["data.result.29_metric","data.result.29_values"]]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df0.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df0.info()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
for key, value in df0["data.result.0_metric"].items():
    
    # The value is understand as a string so we have to cast it as an dict
    
    print(type(value))
    dict = json.loads(value)
    print(type(dict))
    print(dict.keys())
    
        

        
df0['Broker'] = dict.get("hostname")
df0['Disk'] = dict.get("mountpoint")
df0['Data_Center'] = dict.get("dc")
df0 = df0.drop(['data.result.0_metric'], axis=1)

df0.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_val = pd.DataFrame(np.array(df0["data.result.0_values"]), columns=["Timestamp"])
df_val.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_val["Timestamp"].to_numpy()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
ar = np.array(df0["data.result.0_values"])
ar.shape

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_val = pd.DataFrame(np.array(df0["data.result.0_values"]), columns=["Timestamp"])
df_val.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
! install --upgrade pandas

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
#good_columns = dataiku.Dataset("good_columns")
#good_columns.write_with_schema(df)