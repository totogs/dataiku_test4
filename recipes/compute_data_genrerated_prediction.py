# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
data_generated = dataiku.Dataset("data_generated")
data_generated_df = data_generated.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

data_genrerated_prediction_df = data_generated_df # For this sample code, simply copy input to output


# Write recipe outputs
data_genrerated_prediction = dataiku.Dataset("data_genrerated_prediction")
data_genrerated_prediction.write_with_schema(data_genrerated_prediction_df)
