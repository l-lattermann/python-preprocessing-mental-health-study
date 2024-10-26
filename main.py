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


# Main function
if __name__ == "__main__":
    
    # Load all file paths from json file
    with open('file_paths.json', 'r', encoding='latin1') as changelog:
        file_paths = json.load(changelog)

        data_path = file_paths['data_path'][0]['path']
        changelog_path = file_paths['changelog_path'][0]['path']
        all_changes_path = file_paths['all_changes_path'][0]['path']

    # Initialize the changelog file
    log.init_changelog(changelog_path)

    # Load the study data
    study_data = pd.read_csv(data_path, encoding='utf-8')

    # Write to changelog
    log.write("Loaded study data from: " + data_path)
    
    # Load the changes to be made to the study data
    changes = pd.read_csv(all_changes_path, encoding='utf-8')

    # Write to changelog
    log.write("Loaded changes data from: " + all_changes_path)

    # Check for all non ASCII characters in the column names and data
    # Set to store unique non-ASCII characters
    non_ascii_characters_columns = set()
    non_ascii_characters_data = set()

    # Find non-ASCII characters in column names
    for col in study_data.columns:
        matches = re.findall(r'[^\x00-\x7F]', col)
        non_ascii_characters_columns.update(matches)

    # Find non-ASCII characters in the data
    study_data.applymap(lambda text: non_ascii_characters_data.update(re.findall(r'[^\x00-\x7F]', text)) if isinstance(text, str) else None)

    # Convert characters to hex representation for better readability
    hex_dict_columns = {char: hex(ord(char)) for char in non_ascii_characters_columns}
    hex_dict_data = {char: hex(ord(char)) for char in non_ascii_characters_data}

    # Write to changelog
    log.write("Detected following non-ASCII characters in the column names:", hex_dict_columns)
    log.append("\nDetected following non-ASCII characters in the data:", hex_dict_data)

    # Save columns containing non-breaking spaces to list
    matches_columns_before = study_data.columns[study_data.columns.str.contains(r'[\xa0]', regex=True)].tolist() 

    # Clean non-breaking spaces from the column names
    study_data.columns = [re.sub('\s+', ' ', col) for col in study_data.columns]    # Remove non-breaking spaces

    # Check if Regex was successful
    matches_columns_after = study_data.columns[study_data.columns.str.contains(r'[\xa0]', regex=True)].tolist() 

    # Strip leading and trailing spaces
    study_data.columns = study_data.columns.str.strip()                                     

    # Write to changelog
    log.write("Columns with non-breaking spaces before cleaning:", matches_columns_before)
    log.append("\nColumns with non-breaking spaces after cleaning:", matches_columns_after)

    matches_data_before = []    # List to store the columns with non-breaking spaces
    matches_data_after = []     # List to store the columns with non-breaking spaces
    for column in study_data.columns:
        # Extract data with non-breaking spaces
        matches_data_before.append(study_data[column].apply(lambda x: x if isinstance(x, str) and '\xa0' in x else np.nan).dropna().tolist())   

        # Clean non breaking spaces from the data
        study_data[column] = study_data[column].apply(lambda x: re.sub(r'\s+', '', x) if isinstance(x, str) else x)
        
        # Check if Regex was successful
        matches_data_after.append(study_data[column].apply(lambda x: x if isinstance(x, str) and '\xa0' in x else np.nan).dropna().tolist())
        
        # Strip leading and trailing spaces
        study_data[column] = study_data[column].apply(lambda x: x.strip() if isinstance(x, str) else x)                                         

    # Remove empty lists from matches_data_before and matches_data_after
    matches_data_before = [x for x in matches_data_before if x]
    matches_data_after = [x for x in matches_data_after if x]

    # Write to changelog
    log.write("Cleaned weird spaces in the dataset. Data with non-breaking spaces:", matches_data_before)
    log.append("\nData with non-breaking spaces after cleaning:", matches_data_after)

    # Create a copy for later comparison
    study_data_copy = study_data.copy()    

    # Drop rows with over 50% missing values
    original_count = study_data.shape[0]    # Get the original row count
    study_data = study_data.dropna(thresh=study_data.shape[1]/2)
    drop_lines_percent = (1 - (study_data.shape[0] / original_count)) *100    # Calculate the percentage of dropped rows

    # Write to changelog
    log.write(f"Dropped {drop_lines_percent}% of rows due to missing values.")

    # Apply new shortend column labels
    new_labels_dict = {}    # Dictionary to store the old and new labels
    for old_label in changes['old_labels'].dropna():
        new_labels_dict[old_label] = changes.loc[changes['old_labels'] == old_label, 'new_labels'].values[0]
    study_data.columns = study_data.columns.map(new_labels_dict)

    # Write to changelog
    log.write("Applied new shortened column labels as follows:\n", new_labels_dict)

    # Replace age outliers with NaN
    age_copy = study_data['age'].copy()    # Create a copy for later comparison
    study_data['age'] = study_data['age'].apply(lambda x: np.nan if x < 15 or x > 75 else x)    # Age thresholds were manually selected from excel data
    age_checksum = (study_data['age'] != age_copy).sum()    # Check how many rows were changed
    
    # Write to changelog
    log.write(f"Replaced age outliers with NaN for {age_checksum} rows.")

    # Fill missing age values with the median
    age_copy_2 = study_data['age'].copy()    # Create a copy for later comparison
    age_median = study_data['age'].median()
    study_data.fillna({'age': age_median}, inplace=True)
    age_checksum_2 = (age_copy_2 != study_data['age']).sum()    # Check how many rows were changed

    # Write to changelog
    log.write(f"Filled missing age values with the median: {age_median}  for {age_checksum_2} rows.")

    # Get the encoding type for the columns
    encoding_types_dict = dict(zip(changes['new_labels'], changes['encoding_type']))

    # Write to changelog
    log.write("Encoding types for columns are as follows:", encoding_types_dict)

    # Convert JSON-like strings in 'encoding' column in changes df to actual python dict
    temp_store_encoding = changes['encoding'].copy()    # Create a copy for later comparison
    for encoding_type in changes['encoding_type']:
        if encoding_type in {'bin', 'ord'}:
            changes['encoding'] = changes['encoding'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    # Get non-converted rows
    non_converted_rows = changes.loc[changes['encoding'].apply(lambda x: isinstance(x, str))].index
    # Write to changelog
    bin_ord_checksum = (changes['encoding'] != temp_store_encoding).sum()    # Check how many rows were changed
    log.write(f"Converted JSON-like strings in 'encoding' column to actual python dict for {bin_ord_checksum} rows.\nRows that were not converted:", non_converted_rows)

    # Apply binary and ordinal encoding to the respective columns
    bin_ord_applied_succ = []       # List to store the columns that binary and ordinal encoding was applied to
    bin_ord_applied_fail = []       # List to store the columns that binary and ordinal encoding was not applied to
    column_prob_values_dict = {}    # Dictionary to store the columns with problematic values
    for column_name, encoding_type in encoding_types_dict.items():
        if encoding_type in {'bin', 'ord'}:

            # Get the encoding dictionary from the changes dataframe
            if isinstance(changes.loc[changes['new_labels'] == column_name, 'encoding'].values[0], dict):
                encoding = changes.loc[changes['new_labels'] == column_name, 'encoding'].values[0]
                
                # Map the study data to the encoding
                study_data[column_name] = study_data[column_name].map(encoding)

                # Check if encoding was successful
                unique_values = set(study_data[column_name].unique()) | {np.nan}
                expected_values = set(encoding.values()) | {np.nan}
                if unique_values == expected_values:
                    bin_ord_applied_succ.append(column_name)
                else:
                    bin_ord_applied_fail.append(column_name)
                    log.write(f"Binary and ordinal encoding was not successful for column: {column_name}\n the following are the unique values: {unique_values}\n the following are the expected values: {expected_values}")
                    
                    # Get the problematic colums where encoding was not successful
                    column_prob_values_dict[column_name] = set(study_data[column_name].unique()) - (set(encoding.values()) | {np.nan})

    # Write to changelog
    log.write("Applied binary and ordinal encoding to the following columns successfully:", bin_ord_applied_succ)
    log.write("Applied binary and ordinal encoding to the following columns not successfully:", bin_ord_applied_fail)
    log.write("Columns with problematic values after encoding:", column_prob_values_dict)
        
    
    # Get a list of columns with mostly missing values
    missing_thresh = 0.999    # Set the threshold for missing values
    missing_values_columns = study_data.columns[study_data.isna().mean() > missing_thresh].tolist()

    # Write to changelog
    log.write(f"Columns with mostly missing values (Threshold = {missing_thresh}):", missing_values_columns)
        

    # Fill missing values with the mean of the column
    columns_fillna_applied_list = []
    columns_fillna_not_applied_list = []
    for column_name in changes['new_labels'].dropna():
        
        encoding_type = changes.loc[changes['new_labels'] == column_name, 'encoding_type'].values[0]
        print("encoding type is: ", encoding_type)

        # Missing values in binary columns get filled with a probabilistic approach to preserve the overall mean
        if changes.loc[changes['new_labels'] == column_name, 'encoding_type'].values[0] == 'bin':

            # Calculate the proportion of 1s and 0s in the binary column
            # Check if the column has only 1s and 0s
            uniques = study_data[column_name].dropna().unique()
            log.write(f"Trying to perform numeric action on string.. Search Unique values in column {column_name} for strings: ", uniques)
            proportion_of_1s = study_data[column_name].mean()  # Fraction of 1s (mean of binary data)

            # Generate random choices of 0 or 1 based on the existing proportion
            missing_values_count = study_data[column_name].isna().sum()
            fill_values = np.random.choice([0, 1], size=missing_values_count, p=[1 - proportion_of_1s, proportion_of_1s])

            # Fill missing values with the generated random choices
            study_data.loc[study_data[column_name].isna(), column_name] = fill_values

            # Append the column name to the list of columns with filled missing values
            columns_fillna_applied_list.append(column_name)

        # Missing values in ordinal columns get filled with the mode of the column to preserve the overall mean
        elif changes.loc[changes['new_labels'] == column_name, 'encoding_type'].values[0] == 'ord':
            study_data[column_name].fillna(study_data[column_name].mode()[0], inplace=True)

            # Append the column name to the list of columns with filled missing values
            columns_fillna_applied_list.append(column_name)

        # Missing values in one-hot columns get filled with NaN
        elif changes.loc[changes['new_labels'] == column_name, 'encoding_type'].values[0] == 'one-hot':
            study_data[column_name].fillna(np.nan, inplace=True)

            # Append the column name to the list of columns with filled missing values
            columns_fillna_applied_list.append(column_name)

        # Missing values in TF-IDF columns get filled with empty string
        elif changes.loc[changes['new_labels'] == column_name, 'encoding_type'].values[0] == 'TF-IDF':
            study_data[column_name].fillna('', inplace=True)

            # Append the column name to the list of columns with filled missing values
            columns_fillna_applied_list.append(column_name)

        else:
                log.write("No encoding type found for column: " + column_name)
                # Append the column name to the list of columns not filled missing values
                columns_fillna_not_applied_list.append(column_name)
                continue
        
    # Write to changelog:
    log.write("Filled missing values in the following columns: ", columns_fillna_applied_list)
    log.write("No fillings were applied in following columns: ", columns_fillna_not_applied_list)
    
    # Apply TF-IDF encoding to the contextual columns
    tfidf_applied_list = []    # List to store the columns that TF-IDF encoding was applied to
    for column_name, encoding_type in encoding_types_dict:
        if encoding_type == 'TF-IDF':
            # Apply TF-IDF encoding
            tfidf = sk.feature_extraction.text.TfidfVectorizer()
            study_data[column_name] = tfidf.fit_transform(study_data[column_name])

            # Append the label to the list of applied TF-IDF encodings
            tfidf_applied_list.append(column_name)

    # Write to changelog
    log.write("Applied TF-IDF encoding to the following columns:", tfidf_applied_list)
    
    # Apply one-hot encoding to the multiple choice columns
    one_hot_applied_list = []    # List to store the columns that one-hot encoding was applied to
    for column_name, encoding_type in encoding_types_dict:
        if encoding_type == 'one-hot':

            # Initialize the OneHotEncoder
            encoder = sk.preprocessing.OneHotEncoder(sparse_output=False, handle_unknown='ignore')

            # Apply one-hot encoding to the specified column
            one_hot_encoded = encoder.fit_transform(study_data[[column_name]])

            # Convert to DataFrame and rename columns with the original label prefix
            one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out([column_name]))

            # Concatenate the new one-hot encoded DataFrame with the original DataFrame
            study_data = pd.concat([study_data.drop(columns=[column_name]), one_hot_df], axis=1)

            # Append the label to the list of applied one-hot encodings
            one_hot_applied_list.append(column_name)
    
    # Write to changelog
    log.write("Applied one-hot encoding to the following columns:", one_hot_applied_list)

    

            ## fill bin and ord values before one hot and idf encoding. and try to fill one hot and idf missing vlaues during encodign