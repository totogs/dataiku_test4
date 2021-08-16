# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
disk_usage = dataiku.Dataset("disk_usage")
disk_usage_df = disk_usage.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

json_df = disk_usage_df # For this sample code, simply copy input to output


# Write recipe outputs
json = dataiku.Dataset("Json")
json.write_with_schema(json_df)
