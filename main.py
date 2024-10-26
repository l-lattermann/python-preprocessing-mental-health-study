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


# Main function
if __name__ == "__main__":
    
    # Load all file paths from json file
    with open('file_paths.json', 'r', encoding='latin1') as changelog:
        file_paths = json.load(changelog)

        data_path = file_paths['data_path'][0]['path']
        changelog_path = file_paths['changelog_path'][0]['path']
        all_changes_path = file_paths['all_changes_path'][0]['path']

    # Initialize the changelog file
    changelog = log.Changelog(changelog_path)

    # Load the study data
    study_data = pd.read_csv(data_path, encoding='utf-8')
    changelog.write("Loaded study data from: " + data_path)
    
    # Load the changes to be made to the study data
    changes = pd.read_csv(all_changes_path, encoding='utf-8')
    changelog.write("Loaded changes data from: " + all_changes_path)

    # Create a copy for later comparison
    study_data_copy = study_data.copy()    

    # Remove unnecessary characters from the column names and data
    dc.remove_unecesarry_characters(study_data, changelog)

    # Drop missing values
    dc.drop_missing_values(study_data, changelog)

    # Apply shortened labels
    dc.apply_shortened_labels(study_data, changes, changelog)

    # Replace age outliers with NaN
    dc.replace_age_outliers(study_data, changelog)

    # Fill missing age values with the median
    enc.fill_missing_values(study_data, changes, changelog)

    # Apply binary and ordinal encoding
    enc.apply_bin_ord_encoding(study_data, changes, changelog)

