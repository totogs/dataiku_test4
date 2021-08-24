# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu



# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

data_generated_df = ... # Compute a Pandas dataframe to write into data_generated


# Write recipe outputs
data_generated = dataiku.Dataset("data_generated")
data_generated.write_with_schema(data_generated_df)
