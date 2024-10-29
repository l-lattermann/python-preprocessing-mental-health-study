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
    enc.fill_missing_age_values(study_data, changes, changelog)

    # Apply binary and ordinal encoding
    enc.apply_bin_ord_encoding(study_data, changes, changelog)

    # Fill the rest of the missing values
    enc.fill_missing_values(study_data, changes, changelog)

    # Check for missing values
    enc.check_for_missing_values(study_data, changelog)

    # Apply TF-IDF encoding
    enc.tfidf_encoding(study_data, changes, changelog)

    # Apply one-hot encoding
    enc.one_hot_encoding(study_data, changes, changelog)

    # ======================================================================== CLUSTERING ========================================================================

