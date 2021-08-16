# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
json = dataiku.Dataset("Json")
df = json.get_dataframe()
df.head(10)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df=pd.pivot_table(df,index=['shopCode','Product'],columns=df.groupby(['shopCode','Product']).cumcount().add(1),values=['Code','Score'],aggfunc='sum')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
pivot = dataiku.Dataset("Pivot")
pivot.write_with_schema(df)