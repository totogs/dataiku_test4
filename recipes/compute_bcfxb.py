# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
good_columns_prepared = dataiku.Dataset("good_columns_prepared")
good_columns_prepared_df = good_columns_prepared.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

bcfxb_df = good_columns_prepared_df # For this sample code, simply copy input to output


# Write recipe outputs
bcfxb = dataiku.Dataset("bcfxb")
bcfxb.write_with_schema(bcfxb_df)
