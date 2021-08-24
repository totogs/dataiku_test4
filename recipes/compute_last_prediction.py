# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
json_prepared_filtered = dataiku.Dataset("Json_prepared_filtered")
json_prepared_filtered_df = json_prepared_filtered.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

last_prediction_df = json_prepared_filtered_df # For this sample code, simply copy input to output


# Write recipe outputs
last_prediction = dataiku.Dataset("last_prediction")
last_prediction.write_with_schema(last_prediction_df)
