# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
good_columns_prepared = dataiku.Dataset("good_columns_prepared")
df = good_columns_prepared.get_dataframe()
df.head(5)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df["data.result.0_values"] = list(df["data.result.0_values"])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df["data.result.0_values"][0]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
test = dataiku.Dataset("test")
test.write_with_schema(df)