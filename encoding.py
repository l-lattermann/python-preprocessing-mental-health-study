import pandas as pd
import numpy as np
import json
import sklearn as sk




def _get_encoding_type(study_data, changes, logger: object) -> dict:    
    """
    Get the encoding type for a column from the changes DataFrame.

    Parameters
    ----------
    study_data : pd.DataFrame
        The name of the column to get the encoding type for.
    changes : pd.DataFrame
        The DataFrame containing the changes to be made to the study data.
    logger : object
        The logger object to write to the changelog.

    Returns
    -------
    dict
        A dictionary containing the column name and the encoding type.
    """

    # Get the encoding type for the columns
    encoding_types_dict = dict(zip(changes['new_labels'], changes['encoding_type']))

    # Write to changelog
    logger.write("Encoding types for columns are as follows:", encoding_types_dict)

    # Convert JSON-like strings in 'encoding' column in changes df to actual python dict
    temp_store_encoding = changes['encoding'].copy()    # Create a copy for later comparison
    for encoding_type in changes['encoding_type']:
        if encoding_type in {'bin', 'ord'}:
            changes['encoding'] = changes['encoding'].map(lambda x: json.loads(x) if isinstance(x, str) else x)
    # Get non-converted rows
    non_converted_rows = changes.loc[changes['encoding'].apply(lambda x: isinstance(x, str))].index
    # Write to changelog
    bin_ord_checksum = (changes['encoding'] != temp_store_encoding).sum()    # Check how many rows were changed
    logger.write(f"Converted JSON-like strings in 'encoding' column to actual python dict for {bin_ord_checksum} rows.\nRows that were not converted:", non_converted_rows)

    return encoding_types_dict

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

def apply_bin_ord_encoding(study_data: pd.DataFrame, changes: pd.DataFrame, logger: object) -> None:
    """
    Apply binary and ordinal encoding to the study data. According to the encoding dictionary.

    Parameters
    ----------
    study_data : pd.DataFrame
        The DataFrame containing the study data.
    changes : pd.DataFrame
        The DataFrame containing the changes to be made to the study data.
    encoding_dict : dict
        A dictionary containing the column name and the encoding type.
    logger : object
        The logger object to write to the changelog.

    Returns
    -------
    None
    """

    # Get the encoding dictionary
    encoding_dict = _get_encoding_type(study_data, changes, logger)

    bin_ord_applied_succ = []       # List to store the columns that binary and ordinal encoding was applied to
    bin_ord_applied_fail = []       # List to store the columns that binary and ordinal encoding was not applied to
    column_prob_values_dict = {}    # Dictionary to store the columns with problematic values
    for column_name, encoding_type in encoding_dict.items():
        if encoding_type in {'bin', 'ord'}:

            # Get the encoding dictionary from the changes dataframe
            if isinstance(changes.loc[changes['new_labels'] == column_name, 'encoding'].values[0], dict):
                encoding = changes.loc[changes['new_labels'] == column_name, 'encoding'].values[0]
                
                # Unique values in the column before encoding
                unique_values_before = set(study_data[column_name].unique()) | {np.nan}

                # Map the study data to the encoding
                study_data[column_name] = study_data[column_name].map(encoding)

                # Check if encoding was successful
                unique_values = set(study_data[column_name].unique()) | {np.nan}
                expected_values = set(encoding.values()) | {np.nan}
                if unique_values == expected_values:
                    bin_ord_applied_succ.append(column_name)
                else:
                    bin_ord_applied_fail.append(column_name)
                    logger.write(f"Binary and ordinal encoding was not successful for column: {column_name}\n \
                                 the following are the unique values: {unique_values}\n  \
                                 the following are the unique values before encoding: {unique_values_before}\n \
                                 the following are the expected values: {expected_values}")
                    
                    # Get the problematic colums where encoding was not successful
                    column_prob_values_dict[column_name] = set(study_data[column_name].unique()) - (set(encoding.values()) | {np.nan})

    logger.write("Applied binary and ordinal encoding to the following columns successfully:", bin_ord_applied_succ)        # Write to changelog
    logger.write("Applied binary and ordinal encoding to the following columns not successfully:", bin_ord_applied_fail)
    logger.write("Columns with problematic values after encoding:", column_prob_values_dict)
        
    
    # Get a list of columns with mostly missing values
    missing_thresh = 0.999                                                                                      # Set the threshold for missing values
    missing_values_columns = study_data.columns[study_data.isna().mean() > missing_thresh].tolist()             # Get the columns with mostly missing values  

    logger.write(f"Columns with mostly missing values (Threshold = {missing_thresh}):", missing_values_columns) # Write to changelog
    
    return

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
            study_data[column_name].fillna(np.nan, inplace=True)                        # Fill missing values with NaN
            columns_fillna_applied_list.append(column_name)                             # Append column name to check list

        # Fill TF-IDF encoded columns with an empty string
        elif encoding_type == 'TF-IDF':
            study_data[column_name].fillna('', inplace=True)                            # Fill missing values with an empty string
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

def tfidf_encoding(study_data: pd.DataFrame, changes: pd.DataFrame, logger: object) -> None:
        """
        Apply TF-IDF encoding to the study data.

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
        # Get the encoding dictionary
        encoding_dict = _get_encoding_type(study_data, changes, logger)

        tfidf_applied_list = []    # List to store the columns that TF-IDF encoding was applied to
        for column_name, encoding_type in encoding_dict.items():
            if encoding_type == 'TF-IDF':
                # Apply TF-IDF encoding
                tfidf = sk.feature_extraction.text.TfidfVectorizer()
                study_data[column_name] = tfidf.fit_transform(study_data[column_name])

                # Append the label to the list of applied TF-IDF encodings
                tfidf_applied_list.append(column_name)

        # Write to changelog
        logger.write("Applied TF-IDF encoding to the following columns:", tfidf_applied_list)
        
        return

def one_hot_encoding(study_data: pd.DataFrame, changes: pd.DataFrame, logger: object) -> None:
        """
        Apply one-hot encoding to the study data.

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

        # Get the encoding dictionary
        encoding_dict = _get_encoding_type(study_data, changes, logger)

        one_hot_applied_list = []    # List to store the columns that one-hot encoding was applied to
        for column_name, encoding_type in encoding_dict:
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
        logger.write("Applied one-hot encoding to the following columns:", one_hot_applied_list)

        

                ## fill bin and ord values before one hot and idf encoding. and try to fill one hot and idf missing vlaues during encodign