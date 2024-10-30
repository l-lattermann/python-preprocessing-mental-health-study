import pandas as pd
import numpy as np
import re
from Levenshtein import ratio

def remove_unecesarry_characters(study_data: pd.DataFrame, logger: object) -> None:
    """
    Removes unnecessary characters from the column names and data.

    Parameters
    ----------
    study_data : pd.DataFrame
        The DataFrame containing the study data.

    logger : object
        The logger object to write to the changelog.

    """
    # Find non-ASCII characters in column names
    non_ascii_characters_columns = set()    # Set to store non-ASCII characters in the column names
    non_ascii_characters_data = set()       # Set to store non-ASCII characters in the data

    for col in study_data.columns:
        matches = re.findall(r'[^\x00-\x7F]', col)      # Find non-ASCII characters in the column names
        non_ascii_characters_columns.update(matches)    # Add non-ASCII characters to the set

    for col in study_data.columns:
        study_data[col].map(lambda text: non_ascii_characters_data.update(
        re.findall(r'[^\x00-\x7F]', text)) if isinstance(text, str) else None)    # Find non-ASCII characters in the data

    hex_dict_columns = {char: hex(ord(char)) for char in non_ascii_characters_columns}  # Create dictionary from non ASCII characters with hexcode
    hex_dict_data = {char: hex(ord(char)) for char in non_ascii_characters_data}        
    
    logger.write("Detected following non-ASCII characters in the column names:", hex_dict_columns)  # Write to changelog
    logger.append("\n\n\nDetected following non-ASCII characters in the data:", hex_dict_data)

    # Remove non-ASCII characters from the column names
    matches_columns_before = study_data.columns[study_data.columns.str.contains(r'[\xa0]', regex=True)].tolist()   # Find non-breaking spaces
    study_data.columns = [re.sub(r'\s+', ' ', col) for col in study_data.columns]                                  # Remove non-breaking spaces
    matches_columns_after = study_data.columns[study_data.columns.str.contains(r'[\xa0]', regex=True)].tolist()    # Find non-breaking spaces after cleaning 
    study_data.columns = study_data.columns.str.strip()    # Strip leading and trailing spaces

    logger.write("Columns with non-breaking spaces before cleaning:", matches_columns_before)       # Write to changelog
    logger.append("\n\n\nColumns with non-breaking spaces after cleaning:", matches_columns_after)

    # Remove non-ASCII characters from the data
    matches_data_before = []    # List to store data with non-breaking spaces before cleaning
    matches_data_after = []     # List to store data with non-breaking spaces after cleaning
    for column in study_data.columns:
        matches_data_before.append(study_data[column].apply(
            lambda x: x if isinstance(x, str) and '\xa0' in x else np.nan   # Find non-breaking spaces before cleaning
            ).dropna().tolist())   

        study_data[column] = study_data[column].apply(
            lambda x: re.sub(r'\s+', ' ', x) if isinstance(x, str) else x)  # Remove non-breaking spaces
        
        matches_data_after.append(study_data[column].apply(
            lambda x: x if isinstance(x, str) and '\xa0' in x else np.nan   # Find non-breaking spaces after cleaning
            ).dropna().tolist())
        study_data[column] = study_data[column].apply(
            lambda x: x.strip() if isinstance(x, str) else x)               # Strip leading and trailing spaces                                         

    matches_data_before = [x for x in matches_data_before if x]             # Remove empty list elements
    matches_data_after = [x for x in matches_data_after if x]              

    logger.write("Cleaned weird spaces in the dataset. Data with non-breaking spaces:", matches_data_before)   # Write to changelog
    logger.append("\n\nData with non-breaking spaces after cleaning:", matches_data_after)

# Drop rows with over 50% missing values
def drop_missing_values(study_data: pd.DataFrame, logger: object) -> None:
    """
    Drops rows with over 50% missing values from the study data.

    Parameters
    ----------
    study_data : pd.DataFrame
        The DataFrame containing the study data.

    Returns
    -------
    None

    """
    original_count = study_data.shape[0]                                            # Get the original row count
    study_data = study_data.dropna(thresh=study_data.shape[1]/2)                    # Drop rows with over 50% missing values
    drop_lines_percent = (1 - (study_data.shape[0] / original_count)) *100          # Calculate the percentage of dropped rows

    logger.write(f"Dropped {drop_lines_percent}% of rows due to missing values.")   # Write to changelog

# Apply new shortend column labels
def apply_shortened_labels(study_data: pd.DataFrame, changes: pd.DataFrame, logger: object) -> None:
    """
    Applies new shortened column labels to the study data.

    Parameters
    ----------
    study_data : pd.DataFrame
        The DataFrame containing the study data.
    changes : pd.DataFrame
        The DataFrame containing the changes to be made to the study data.

    Returns
    -------
    None

    """
    new_labels_dict = {}                                                                # Dictionary to store the old and new labels
    for old_label in changes['old_labels'].dropna():
        new_labels_dict[old_label] = changes.loc[
            changes['old_labels'] == old_label, 'new_labels'                            # Create a dictionary with old and new labels
            ].values[0]
    study_data.columns = study_data.columns.map(new_labels_dict)                        # Apply new shortened column labels

    logger.write("Applied new shortened column labels as follows:\n", new_labels_dict)  # Write to changelog

def replace_age_outliers(study_data: pd.DataFrame, logger: object) -> None:
    """
    Replaces age outliers with NaN and fills missing age values with the median.

    Parameters
    ----------
    study_data : pd.DataFrame
        The DataFrame containing the study data.

    Returns
    -------
    None

    """
    # Replace age outliers with NaN
    age_copy = study_data['age'].copy()                     # Create a copy for later comparison
    study_data['age'] = study_data['age'].apply(
        lambda x: np.nan if x < 15 or x > 75 else x         # Replace age outliers with NaN
        )
    age_checksum = (study_data['age'] != age_copy).sum()    # Check how many rows were changed

    logger.write(f"Replaced age outliers with NaN for {age_checksum} rows.") # Write to changelog

def fill_missing_age_values(study_data, changes, logger):
    """
    Fill missing values in the study data with the median.

    Parameters
    ----------
    study_data : pd.DataFrame
        The DataFrame containing the study data.
    changes : pd.DataFrame
        The DataFrame containing the changes to be made to the study data.
    logger : object 
        The logger object to write to the changelog.

    Returns
    -------
    None
    """

    age_copy_2 = study_data['age'].copy()    # Create a copy for later comparison
    age_median = study_data['age'].median()
    study_data.fillna({'age': age_median}, inplace=True)
    age_checksum_2 = (age_copy_2 != study_data['age']).sum()    # Check how many rows were changed

    # Write to changelog
    logger.write(f"Filled missing age values with the median: {age_median}  for {age_checksum_2} rows.")

def fill_missing_values(study_data: pd.DataFrame, changes: pd.DataFrame, logger: object) -> None:
    """
    Fill missing values in the study data according to the encoding type.

    Parameters
    ----------
    study_data : pd.DataFrame
        The DataFrame containing the study data.
    changes : pd.DataFrame
        The DataFrame containing the changes to be made to the study data.
    logger : object
        The logger object to write to the changelog.

    Returns
    -------
    None

    """

    columns_fillna_applied_list = []                                                 # List to store the columns with filled missing values
    columns_fillna_not_applied_list = []                                             # List to store the columns not filled missing values
    for column_name in changes['new_labels'].dropna():
        
        encoding_type = changes.loc[
                changes['new_labels'] == column_name, 'encoding_type'                   # Get the encoding type for the column
                ].values[0]   

        # Fill binary with probabilistic approach to preserve the overall mean
        if encoding_type == 'bin':
            if set(study_data[column_name].dropna()) == {"0", "1"}:                  # Check if the column is binary
                study_data[column_name] = study_data[column_name].astype("Int64")    # Convert the column to integer
            try:
                proportion_of_1s = study_data[column_name].mean()                    # Fraction of 1s (mean of binary data)

            except TypeError:                                                        # Log data types if not binary
                dtype_list = []                                                      # List to store the data types of the column
                for cell in study_data[column_name]:
                    dtype_list.append(type(cell))                                    # Get the data types of the column
                    dtype_list = list(set(dtype_list))                               # Get the unique data types in the column
                logger.write(f"Column {column_name} has hast dtype: ", dtype_list)   # Write to changelog
                continue

            missing_values_count = study_data[column_name].isna().sum()                 # Count the missing values in the column
            fill_values = np.random.choice(                                             # Generate random values
                    [0, 1],                                                             # Set to choose from
                    size=missing_values_count,                                          # Number of missing values
                    p=[1 - proportion_of_1s, proportion_of_1s]                          # Proportions of 1 and 0
                    )

            study_data.loc[study_data[column_name].isna(), column_name] = fill_values   # Fill up the missing values from the list
            columns_fillna_applied_list.append(column_name)                             # Append column name to check list

        # Fill ordinal with the mode value to preserve the overall mode
        elif encoding_type == 'ord':
            try:
                study_data[column_name] = study_data[column_name].fillna(
                        study_data[column_name].mode()[0]                               # Fill missing values with mode
                        )
            except KeyError:
                logger.write(f"Column {column_name} has no mode value.")                # Write to changelog

            columns_fillna_applied_list.append(column_name)                             # Append column name to check list

        # Fill one-hot encoded columns with NaN
        elif encoding_type == 'one-hot':
            study_data.fillna({column_name: ''}, inplace=True)                     # Fill missing values with NaN
            columns_fillna_applied_list.append(column_name)                             # Append column name to check list

        # Fill TF-IDF encoded columns with an empty string
        elif encoding_type == 'TF-IDF':
            study_data.fillna({column_name: ''}, inplace=True)                            # Fill missing values with an empty string
            columns_fillna_applied_list.append(column_name)                             # Append column name to check list

        else:
                logger.write("No encoding type found for column: " + column_name)       # Write to changelog
    
    # Get the columns where encoding was not applied
    columns_fillna_not_applied_list = list(
            set(study_data.columns) - set(columns_fillna_applied_list)                   # Set comprehension for performance
            )

    logger.write("Filled missing values in the following columns: ", columns_fillna_applied_list)       # Write to changelog
    logger.write("No fillings were applied in following columns: ", columns_fillna_not_applied_list)

    return

def check_for_missing_values(study_data: pd.DataFrame, logger: object) -> None:
    """
    Check for missing values in the study data.

    Parameters
    ----------
    study_data : pd.DataFrame
        The DataFrame containing the study data.
    logger : object
        The logger object to write to the changelog.

    Returns
    -------
    None

    """

    missing_values_columns = study_data.columns[study_data.isna().any()].tolist()    # Get the columns with missing values
    logger.write("Columns with missing values:", missing_values_columns)             # Write to changelog

    return

def levenshtine_distance(string_list: list, threshold: float, logger: object) -> dict:
    """
    Calculates the Levenshtein distance between strings in a list.

    Parameters
    ----------
    string_list : list
        The list of strings to calculate the Levenshtein distance.

    Returns
    -------
    dict
        The dictionary containing the Levenshtein distance between strings.

    """
    lev_frame= pd.DataFrame(columns=['string1', 'string2', 'similarity'])    # Dictionary to store the Levenshtein distance
    for i, string1 in enumerate(string_list):
        for string2 in string_list[i+1:]:
            similarity = ratio(string1, string2)    # Calculate the Levenshtein distance
            if similarity > threshold:
                new_row = pd.DataFrame([[string1, string2, similarity]], columns=lev_frame.columns)
                lev_frame = pd.concat([lev_frame, new_row], ignore_index=True)

    lev_frame.sort_values(by='similarity', ascending=False, inplace=True)    # Sort the dictionary by similarity    
                
    logger.write("Calculated Levenshtein distance between strings:", lev_frame)  # Write to changelog
    return lev_frame 

def lower_case(study_data: pd.DataFrame, logger: object) -> None:
    """
    Converts all strings in the DataFrame to lowercase.

    Parameters
    ----------
    study_data : pd.DataFrame
        The DataFrame containing the study data.

    Returns
    -------
    None

    """
    for col in study_data.columns:
        study_data[col] = study_data[col].map(
            lambda x: x.lower() if isinstance(x, str) else x        # Convert all strings to lowercase
            )    
    logger.write("Converted all strings to lowercase.")             # Write to changelog


def replace_similiar_strings(study_data: pd.DataFrame, similars_df: pd.DataFrame, logger: object) -> None:
    """
    Replaces similar strings in the DataFrame.

    Parameters
    ----------
    study_data : pd.DataFrame
        The DataFrame containing the study data.
    similars_df : pd.DataFrame
        The DataFrame containing the similar strings.
    logger : object
        The logger object to write to the changelog.

    Returns
    -------
    None

    """
    replace_dict = {}    # Dictionary to store the similar strings
    for value1, value2 in zip(similars_df['string1'], similars_df['string2']):
        count1 = (study_data == value1).sum().sum()
        count2 = (study_data == value2).sum().sum()

        if count1 > count2:
            study_data.replace(value2, value1, inplace=True)
            replace_dict[value2] = value1
        else:
            study_data.replace(value1, value2, inplace=True)
            replace_dict[value1] = value2
        

    logger.write("Replaced similar strings in the dataset.", replace_dict)               # Write to changelog

    return 