
import json
from pathlib import Path
import inspect
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
import sklearn as sk
import re

import changelog as log
import data_cleaning as dc
import encoding as enc
import data_io as io


# Main function
if __name__ == "__main__":

    # ===================================================================== LOGGER =====================================================================
    # Initialize the changelog file
    changelog = log.Changelog(JSON_path= True)

    # ===================================================================== LOAD DATA =====================================================================
    # Load data 
    study_data = io.load_data(changelog)
    study_data.head()
    
    # Load the changes to be made to the study data
    changes = io.load_changes(changelog)

    # Create a copy for later comparison
    study_data_copy = study_data.copy()

    # ===================================================================== DATA CLEANING ===================================================================== 

    # Remove unnecessary characters from the column names and data
    dc.remove_unecesarry_characters(study_data, changelog)

    # Drop missing values
    dc.drop_missing_values(study_data, changelog)

    # Apply shortened labels
    dc.apply_shortened_labels(study_data, changes, changelog)

    # Replace age outliers with NaN
    dc.replace_age_outliers(study_data, changelog)

    # ======================================================================== ENCODING ========================================================================

    # Fill missing age values with the median
    dc.fill_missing_age_values(study_data, changes, changelog)

    # Apply binary and ordinal encoding
    enc.apply_bin_ord_encoding(study_data, changes, changelog)

    # Fill the rest of the missing values
    dc.fill_missing_values(study_data, changes, changelog)

    # Check for missing values
    dc.check_for_missing_values(study_data, changelog)

    # Convert all strings to lowercase
    dc.lower_case(study_data, changelog)

    # Log the unique values for one hot encoding
    uniques  = enc.get_onehot_unique_values(study_data, changes, changelog)

    # Catch typos and similars in one hot data
    similars = dc.levenshtine_distance(uniques, 0.9, changelog)

    # Replace similars with the more frequent value
    dc.replace_similiar_strings(study_data, similars, changelog)

    # Apply one hot encoding
    study_data = enc.one_hot_encoding(study_data, changes, changelog)

    # Check for any remaining float integers and replace them with ints
    dc.int_all_floats(study_data, changelog)

    # Save the cleaned and encoded data
    io.save_to_csv(study_data, changelog, 'data/cleaned_and_encoded_data.csv')

    exit()
    # Apply TF-IDF encoding
    enc.tfidf_encoding(study_data, changes, changelog)

    # ======================================================================== CLUSTERING ========================================================================

