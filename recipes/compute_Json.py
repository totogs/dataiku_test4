# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read recipe inputs
disk_usage = dataiku.Dataset("disk_usage")
df = disk_usage.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df.info()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
json = json.loads(df)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
json = dataiku.Dataset("Json")
json.write_with_schema(df)