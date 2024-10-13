import  pytest
import pandas as pd

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
    

