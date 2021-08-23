# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
json_prepared = dataiku.Dataset("Json_prepared")
json_prepared_df = json_prepared.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

testt_df = json_prepared_df # For this sample code, simply copy input to output


# Write recipe outputs
testt = dataiku.Dataset("testt")
testt.write_with_schema(testt_df)
