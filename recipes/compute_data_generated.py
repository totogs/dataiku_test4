# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu


# Read recipe inputs
json_prepared = dataiku.Dataset("json_stacked_distinct")
df = json_prepared.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from datetime import timedelta
from datetime import datetime
import random

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
max_index = len(df)-10
random_index = random.randint(0,max_index)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_generated = df[random_index:random_index+10]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
actualdate = df_generated["date"].max()
listdate = []
for i in range(len(df_generated)):
    actualdate = actualdate + timedelta(minutes=15)
    listdate.append(actualdate)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
listdate.reverse()
df_generated["date"] = listdate

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_generated

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.


# Write recipe outputs
data_generated = dataiku.Dataset("data_generated")
data_generated.write_with_schema(df_generated)