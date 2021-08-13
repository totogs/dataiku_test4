# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
prepared = dataiku.Dataset("prepared")
prepared_df = prepared.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

goodData_df = prepared_df # For this sample code, simply copy input to output


# Write recipe outputs
goodData = dataiku.Dataset("GoodData")
goodData.write_with_schema(goodData_df)
