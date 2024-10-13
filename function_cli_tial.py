import pandas as pd
import numpy as np

import manual_data_preprocessing as mdp


# Create a DataFrame with a column that has three unique values
dataframe = pd.DataFrame({
    'A': ['yes', 'no', 'maybe', 'yes', 'no', None],
    'B': ['yes', 'no', 'maybe', 'yes', 'no', None],
    'C': ['yes', 'no', 'maybe', 'yes', 'no', None]
})



# Binary encode the column
binary_encoded_dataframe = mdp.binary_encoding_yes_no_maybe(dataframe)

# Create the expected DataFrame
expected_dataframe = pd.DataFrame({
    'A_yes': [1, 0, 0.5, 1, 0, None],
    'B_yes': [1, 0, 0.5, 1, 0, None],
    'C_yes': [1, 0, 0.5, 1, 0, None]
})

print("final result")
print( "binary_encoded_dataframe: ", binary_encoded_dataframe.to_string())
print( "expected_dataframe: ", expected_dataframe.to_string())
input("Press Enter to continue...") 