import pandas as pd
import numpy as np
import re



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

    study_data.applymap(lambda text: non_ascii_characters_data.update(
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

    

