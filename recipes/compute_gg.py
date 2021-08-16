# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
good_columns = dataiku.Dataset("good_columns")
good_columns_df = good_columns.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

gg_df = good_columns_df # For this sample code, simply copy input to output


# Write recipe outputs
gg = dataiku.Dataset("gg")
gg.write_with_schema(gg_df)
