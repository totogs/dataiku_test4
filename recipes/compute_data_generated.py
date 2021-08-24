# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu


# Read recipe inputs
json_prepared = dataiku.Dataset("Json_prepared")
df = json_prepared.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from datetime import timedelta
import random

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
max_index = len(df)-10
random_index = random.randint(0,max_index)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_generated = df[random_index:random_index+10]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
actualdate = datetime
print(lastdate)
for i in range(len(val)):
    lastdate = lastdate + timedelta(minutes=15)
    df.append()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_generated

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

data_generated_df = ... # Compute a Pandas dataframe to write into data_generated


# Write recipe outputs
data_generated = dataiku.Dataset("data_generated")
data_generated.write_with_schema(data_generated_df)