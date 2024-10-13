import  pytest
import pandas as pd
import numpy as np

import manual_data_preprocessing as mdp

def test_impute_missing_numeric_values():
    # Create a DataFrame with missing values
    dataframe = pd.DataFrame({
        'A': [1, 2, 3, None],
        'B': [4, None, 6, 7],
        'C': [8, 9, 10, 11]
    })

    # Impute the missing values
    imputed_dataframe = mdp.impute_missing_numeric_values(dataframe)

    # Create the expected DataFrame
    expected_dataframe = pd.DataFrame({
        'A': [1, 2, 3, 2],
        'B': [4, 5, 6, 7],
        'C': [8, 9, 10, 11]
    })

    # Check that the DataFrames are equal
    pd.testing.assert_frame_equal(imputed_dataframe, expected_dataframe)

def test_binary_encoding_yes_no_maybe():
    # Create a DataFrame with a column that has three unique values
    dataframe = pd.DataFrame({
        'A': ['yes', 'no', 'maybe', 'yes', 'no', None]
    })

    # Binary encode the column
    binary_encoded_dataframe = mdp.binary_encoding_yes_no_maybe(dataframe)

    # Create the expected DataFrame
    expected_dataframe = pd.DataFrame({
        'A': [1, 0, 0.5, 1, 0, np.nan]
    })

    # Print both DataFrames
    print("binary_encoded_dataframe: ", binary_encoded_dataframe.to_string())
    print("expected_dataframe: ", expected_dataframe.to_string())

    # Check that the DataFrames are equal
    pd.testing.assert_frame_equal(binary_encoded_dataframe, expected_dataframe)


