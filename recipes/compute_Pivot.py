# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
json = dataiku.Dataset("Json")
json_df = json.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

pivot_df = json_df # For this sample code, simply copy input to output


# Write recipe outputs
pivot = dataiku.Dataset("Pivot")
pivot.write_with_schema(pivot_df)
