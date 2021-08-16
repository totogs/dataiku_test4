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
# Write recipe outputs
pivot = dataiku.Dataset("Pivot")
pivot.write_with_schema(df)