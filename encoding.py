import pandas as pd
import numpy as np
import json
import sklearn as sk
import Levenshtein as lev




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

def get_onehot_unique_values(study_data: pd.DataFrame, changes: pd.DataFrame, logger: object) -> list:
        """
        Get the unique values in the columns that will be one-hot encoded.

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
        set
            A set containing the unique values in the columns that will be one-hot encoded.

        """

        one_hot_unique_values = []                                                                           # Create unique values list
        for column_name in changes.loc[changes['encoding_type'] == 'one-hot', 'new_labels'].dropna():        # Check if the encoding type is one-hot
                
                study_data[column_name].map(
                    lambda x: one_hot_unique_values.extend(x.split('|')) if isinstance(x, str) else x        # Split the values if they are strings
                    )
                
        one_hot_unique_values = list(set(one_hot_unique_values))                                             # Convert to set to drop duplicates
        logger.write("Unique values in the columns that will be one-hot encoded:", one_hot_unique_values)    # Write to changelog

        return one_hot_unique_values

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

    one_hot_applied_list = []    # List to store the columns that one-hot encoding was applied to
    for column_name in changes.loc[changes['encoding_type'] == 'one-hot', 'new_labels'].dropna():

            study_data[column_name] = study_data[column_name].apply(
                    lambda x: x.split('|') if isinstance(x, str) else x                                    # Split the values in the column
                    )                       

            # Isolate column to encode
            one_hot_column = study_data[column_name]
            
            # Explode lists into single values
            one_hot_column = one_hot_column.explode()
    
            # Perform one-hot encoding
            one_hot = pd.get_dummies(one_hot_column, prefix=column_name).astype(int)

            # Group by the original index and aggregate back to lists
            one_hot_encoded = one_hot.groupby(one_hot.index).max()
            
            # Drop the original column and concatenate one-hot encoded columns
            study_data.drop(column_name, axis=1, inplace=True)

            study_data = study_data.reset_index(drop=True)
            one_hot_encoded = one_hot_encoded.reset_index(drop=True)
            study_data = pd.concat([study_data, one_hot_encoded], axis=1)
            
    logger.write("Applied one-hot encoding to the following columns:", one_hot_applied_list)            # Write to changelog
    logger.append("One-hot df looks like this: ", one_hot.head())                                       # Write to changelog

    return study_data

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
            tfidf_matrix = tfidf.fit_transform(study_data[column_name])

            # Convert the sparse matrix to a DataFrame
            tfidf_df = pd.DataFrame(
                tfidf_matrix.toarray(),
                index=study_data.index,
                columns=tfidf.get_feature_names_out()
            )

            tfidf_df.to_csv(f'data/{column_name}tfidf_matrix.csv')

            # Combine back into the DataFrame if needed
            study_data.drop(column_name, axis=1, inplace=True)
            study_data = pd.concat([study_data, tfidf_df], axis=1)

            # Append the label to the list of applied TF-IDF encodings
            tfidf_applied_list.append(column_name)

    # Write to changelog
    logger.write("Applied TF-IDF encoding to the following columns:", tfidf_applied_list)

    return study_data
