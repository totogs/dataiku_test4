# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read recipe inputs
prepared = dataiku.Dataset("prepared")
df = prepared.get_dataframe()
df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#On fait des sous daframes pour pouvoir faire les modifications sur tout les brokers en memes temps

df0 = df[["data.result.0.metric","data.result.0.values"]]
df1 = df[["data.result.1.metric","data.result.1.values"]]
df2 = df[["data.result.2.metric","data.result.2.values"]]
df3 = df[["data.result.3.metric","data.result.3.values"]]
df4 = df[["data.result.4.metric","data.result.4.values"]]

df5 = df[["data.result.5.metric","data.result.5.values"]]
df6 = df[["data.result.6.metric","data.result.6.values"]]
df7 = df[["data.result.7.metric","data.result.7.values"]]
df8 = df[["data.result.8.metric","data.result.8.values"]]
df9 = df[["data.result.9.metric","data.result.9.values"]]

df10 = df[["data.result.10.metric","data.result.10.values"]]
df11 = df[["data.result.11.metric","data.result.11.values"]]
df12 = df[["data.result.12.metric","data.result.12.values"]]
df13 = df[["data.result.13.metric","data.result.13.values"]]
df14 = df[["data.result.14.metric","data.result.14.values"]]

df15 = df[["data.result.15.metric","data.result.15.values"]]
df16 = df[["data.result.16.metric","data.result.16.values"]]
df17 = df[["data.result.17.metric","data.result.17.values"]]
df18 = df[["data.result.18.metric","data.result.18.values"]]
df19 = df[["data.result.19.metric","data.result.19.values"]]

df20 = df[["data.result.20.metric","data.result.20.values"]]
df21 = df[["data.result.21.metric","data.result.21.values"]]
df22 = df[["data.result.22.metric","data.result.22.values"]]
df23 = df[["data.result.23.metric","data.result.23.values"]]
df24 = df[["data.result.24.metric","data.result.24.values"]]

df25 = df[["data.result.25.metric","data.result.25.values"]]
df26 = df[["data.result.26.metric","data.result.26.values"]]
df27 = df[["data.result.27.metric","data.result.27.values"]]
df28 = df[["data.result.28.metric","data.result.28.values"]]
df29 = df[["data.result.29.metric","data.result.29.values"]]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df0.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
dict = df0["data.result.0.metric"]

for key, value in dict.items():
    
    print(key, '->', dict[key])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
dict = df0["data.result.0.metric"]

for key in dict:
    
    print(key)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
broker
for key, value in df0["data.result.0.metric"].items():
    if key == "hostname":
        print(value)
    else :
        print("non chackal")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df0["data.result.0.metric"].items()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
#good_columns = dataiku.Dataset("good_columns")
#good_columns.write_with_schema(df)