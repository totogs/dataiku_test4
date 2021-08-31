# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
data_prepared = dataiku.Dataset("data_prepared")
data_prepared_df = data_prepared.get_dataframe()


# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

test_df = ... # Compute a Pandas dataframe to write into test
test2_df = ... # Compute a Pandas dataframe to write into test2


# Write recipe outputs
test = dataiku.Dataset("test")
test.write_with_schema(test_df)
test2 = dataiku.Dataset("test2")
test2.write_with_schema(test2_df)
