import json
from pathlib import Path
import inspect
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
import sklearn as sk


# Function to write object to changelog
def write_to_changelog(changelog_path: str, message: str, content: any):
    """
    Writes content to a changelog file.

    Parameters
    ----------
    changelog_path : str
        The path to the changelog file.
    message : str
        The message to be written to the changelog file.
    content : any
        The content to be written to the changelog file.
    """
    # Get the functionname that called write_to_changelog
    calling_function = inspect.stack()[1].function

    with open(changelog_path, 'a', encoding='utf-8') as changelog:
        changelog.write(pd.Timestamp.now().strftime("%Y.%m.%d. %H:%M:1%S ") + "within" + calling_function + ":\n")
        changelog.write("\t" + message + "\n")

            # Check if content is a DataFrame
        if isinstance(content, pd.DataFrame):
            changelog.write("\n" + tabulate(content, headers='keys', tablefmt='pretty'))
        # Check if content is a dictionary
        elif isinstance(content, dict):
            changelog.write("\n" + str(content))
        # Check if content is a list or tuple
        elif isinstance(content, (list, tuple)):
            changelog.write("\n" + ', '.join(map(str, content)))
        # Handle general objects by converting to string if possible
        elif isinstance(content, object):
            if hasattr(content, 'to_string'):
                changelog.write("\n" + str(content.to_string()))
            else:
                changelog.write("\n" + str(content))
        else:
            # Fallback for primitives (int, float, etc.)
            changelog.write("\n" + str(content))
    
        # Close the changelog file
        changelog.close()

# Shorten column labels
def shorten_column_lables(dataframe, changelog_path: str):
    """
    Shortens the column labels of a DataFrame using user input.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame whose column labels should be shortened.
    changelog_path : str
        The path to the changelog file.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with shortened column labels.
    """
    # Create a dictionary to store the new column labels
    new_labels_dict = {}
    # Display the original column label and ask the user for a new label
    for col in dataframe.columns:
        print("\n\nOriginal column label: '{}'".format(col))
        try:
            new_label = input("Enter a new label: ")
            new_labels_dict[col] = new_label
            dataframe.rename(columns={col: new_label}, inplace=True)
        except Exception as e:
            print("An error occurred: ", e)
            continue
    
    # New lables dict to dataframe
    new_labels_df = pd.DataFrame(collumns= ['old lables', 'new lables'], data=[dataframe.columns, new_labels_dict.values()])

    # New lables df to csv
    new_labels_df.to_csv('new_labels.csv', index=False)

    write_to_changelog(changelog_path, "Colum names got change to the following labels: \n", new_labels_df)
    return dataframe

# Extract all unique feature values
def extract_all_unique_feature_values(dataframe):
    """
    Extracts all unique feature values from a DataFrame. 
    Returns a DataFrame containing all unique feature values sorted in ascending order.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame from which to extract unique feature values.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing all unique feature values.

    """
    # Create a dictionary to store unique values for each column
    unique_values_dict = {col: [] for col in dataframe.columns}
    for key in unique_values_dict:
        unique_values_dict[key] = dataframe[key].unique()
    
    # Create a DataFrame from the dictionary
    unique_values_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in unique_values_dict.items()]))

    # Sort the values in each column
    unique_values_df = unique_values_df.apply(lambda col: col.sort_values().values)

    return unique_values_df

# Save DataFrame to CSV
def save_dataframe_to_csv(dataframe, data_path: str, changelog_path: str, name: str):
    """
    Saves a DataFrame to a CSV file.
    Returns None.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame to be saved to a CSV file.
    data_path : str 
        The path where the original data was retrieved.
    changelog_path : str
        The path to the changelog file.
    name : str
        The name of the CSV file.
    """
    name = name + pd.Timestamp.now().strftime("%Y-%m-%d %H-%M-%S")

    # Create new path assuming old path contains file name
    directory = Path(data_path).parent
    new_file_path = directory / f'{name}.csv'

    # Write the DataFrame to a CSV file
    dataframe.to_csv(new_file_path, index=False)

    # Write to changelog
    write_to_changelog(changelog_path, "DataFrame saved to CSV file at:\n{}".format(new_file_path),content= None)

    return None

# Get value counts per feature
def get_value_count_per_feature(dataframe):
    """
    Returns the value counts for each feature in a DataFrame.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame for which to calculate the value counts.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the value counts for each feature.
    """
    # Create a dictionary to store the value counts
    value_counts_dict = {}

    # Iterate over each column and calculate the value counts
    for col in dataframe.columns:
        value_counts_dict[col] = [dataframe[col].count()]  # Store the count as a list for DataFrame construction

    # Convert the dictionary to a DataFrame with a single row
    value_counts_df = pd.DataFrame(value_counts_dict)

    # Transpose the DataFrame for better readability
    value_counts_df = value_counts_df.T

    # Sort the DataFrame by descending value counts
    value_counts_df = value_counts_df.sort_values(by=0, ascending=False)

        
    return value_counts_df

# Extract features with extraordinary value counts
def extract_features_extraodinary_value_counts(dataframe, threshold_over_mean: float):
    """
    Extracts features with extraordinary value counts from a DataFrame.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame from which to extract extraordinary value counts.
    threshold_over_mean : float
        The threshold for the value count to be considered extraordinary.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the features with extraordinary value counts.
    """
    # Calculate the mean value count
    mean_value_count = dataframe.mean()

    # Filter the features with extraordinary value counts
    extraordinary_value_counts = dataframe[dataframe > mean_value_count * threshold_over_mean].dropna()

    return extraordinary_value_counts

# Impute missing values functions
def impute_missing_numeric_values(dataframe):
    """
    Imputes missing values in a DataFrame with the mean of the respective column.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame in which to impute missing values.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with imputed missing values.
    """
    # Impute missing values with the mean of the respective column
    dataframe.fillna(dataframe.mean(), inplace=True)
    print("imputed with mean: ", dataframe.head())

    return dataframe

# Binary encoding for yes no maybe
def binary_encoding_yes_no_maybe(dataframe):
    """
    Encodes 'Yes', 'No', and 'Maybe'/ 'I don't know' as 1, 0, and 0.5, respectively.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame in which to encode 'Yes', 'No', and 'Maybe'.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with encoded values.
    """
    # Create a dictionary to map the values
    encoding_dict = {'Yes': 1, 'yes': 1, 'No': 0, 'no': 0, 'Maybe': 0.5, 'maybe': 0.5,  "I don't know": 0.5}

    # Save the dataframe before encoding
    dataframe_before_encoding = dataframe.copy()

    # Apply the encoding to the DataFrame and save the lables of changed columns in a list
    changed_columns = []
    for x in dataframe.columns:
        
        if dataframe[x].isin(encoding_dict.keys()).any():
            dataframe[x] = dataframe[x].map(encoding_dict)
            changed_columns.append(x)
    
    # Check if the encoding was successful and respective columns only contain 1, 0, or 0.5 or NaN
    if dataframe[changed_columns].isin([1, 0, 0.5, None, np.nan]).all().all():
        return dataframe
    else:
        return dataframe_before_encoding

# ordinal encoding of company size
def ordinal_encoding_company_size(dataframe):
    """
    Encodes the 'company_size' feature using ordinal encoding.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame in which to encode 'company_size'.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with encoded 'company_size'.
    """
    # Create a dictionary to map the values
    encoding_dict = {'1-5': 1, '6-25': 2, '26-100': 3, '100-500': 4, '500-1000': 5, 'More than 1000': 6}

    # Apply the encoding to the DataFrame
    dataframe['company_size'] = dataframe['company_size'].map(encoding_dict)

    return dataframe

# One-hot encoding of categorical features
def one_hot_encoding(dataframe, columns):
    """
    Encodes categorical features using one-hot encoding.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame in which to encode categorical features.
    columns : list
        A list of columns to be one-hot encoded.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with one-hot encoded features.
    """
    # Apply one-hot encoding to the DataFrame
    dataframe = pd.get_dummies(dataframe, columns=columns)

    return dataframe

# Iterate over all columns and ask user to select encoding type from a list
def select_encoding_type(dataframe, changelog_path: str):
    """
    Iterates over all columns in a DataFrame and asks the user to select an encoding type from a list.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame for which to select encoding types.
    changelog_path : str
        The path to the changelog file.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with selected encoding types.
    """
    # Create a dictionary to store the encoding types
    encoding_dict = {}

    # Iterate over each column and ask the user to select an encoding type
    for col in dataframe:
        print("\n\nSelect encoding type for column '{}':".format(col))
        print("Data looks like this: ", dataframe[col].dropna().unique()[:7])
        print("0: Skip column")
        print("1: Binary encoding")
        print("2: Ordinal encoding")
        print("3: One-hot encoding")
        print("4: One-hot encoding, but need speparation of values")
        print("5: TF-IDF encoding")
        print("6: Skip all remaining columns")
        encoding_type = input("Enter the number of the encoding type: ")

        try:
        # Apply the selected encoding type
            if encoding_type == '0':
                continue
            elif encoding_type == '1':
                encoding_dict[col] = 'binary'
            elif encoding_type == '2':
                encoding_dict[col] = 'ordinal'
            elif encoding_type == '3':
                encoding_dict[col] = 'one-hot'
            elif encoding_type == '4':
                encoding_dict[col] = 'one-hot-sep'
            elif encoding_type == '5':
                encoding_dict[col] = 'tf-idf'
            elif encoding_type == '6':
                break
            else:
                print("Invalid encoding type. Please try again.")
        except Exception as e:
            print("An error occurred: ", e)
            continue

    # Write the encoding type to the dictionary to csv
    encoding_dict_df = pd.DataFrame({
        'column': list(encoding_dict.keys()),
        'encoding type': list(encoding_dict.values())
    })
    save_dataframe_to_csv(encoding_dict_df, data_path, changelog_path, 'encoding_dict')
    # Open the changelog file
    write_to_changelog(changelog_path, "Encoding types selected for each column.", encoding_dict_df.to_string())

    return encoding_dict

# Data that contains combinations of predifined standardised textual values will be string parsed and one-hot encoded
def encode_multiple_choice_textual_data(dataframe):
    """
    Encodes multiple choice textual data in a DataFrame using one-hot encoding.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame in which to encode multiple choice textual data.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with encoded multiple choice textual data.
    """
    # Check if column contains pipes
    for col in dataframe.columns:
        if pd.api.types.is_string_dtype(dataframe[col].dropna()):
            if dataframe[col].str.contains('|', regex=False).any():
            # Parse data that contains combinations of predifined standardised textual values from string to list of strings
                string_parsed_df = dataframe[col].map(lambda x: x.split('|') if isinstance(x, str) else x)
                # Check which columns where string parsed
                changelog_dict = {'all cells parsed' : [], 'not all cells parsed' : [], 'no cells parsed' :[]}
                # Iterate over columns
                for column in string_parsed_df.columns:
                    original_col = dataframe[column]
                    parsed_col = string_parsed_df[column]

                    original_col = original_col.dropna()
                    parsed_col = parsed_col.dropna()
                    
                    # Track whether each row was changed or not
                    changes = [isinstance(orig, str) and isinstance(parsed, list) for orig, parsed in zip(original_col, parsed_col)]
                
                    # Check the dtype of the cells in the column
                    if all(changes):  # All values were changed (split)
                        changelog_dict['all cells parsed'].append(column)
                    elif any(changes):  # Some values were changed, some were not
                        changelog_dict['not all cells parsed'].append(column)
                    else:  # No values were changed
                        continue
    # Create a DataFrame from the changelog dictionary
    changelog_df = pd.DataFrame()
    for key, value in changelog_dict.items():
        changelog_df = pd.concat([changelog_df, pd.DataFrame({key: value})], axis=1)

    
    # Write the changelog to the changelog file
    write_to_changelog(changelog_path, "String parsed multiple choice textual data. Columns were changed: ", changelog_df)

    return string_parsed_df

# Main function
if __name__ == "__main__":
    # Load relevant file paths from file_paths.json
    with open('file_paths.json', 'r', encoding='utf-8') as file:
        file_paths = json.load(file)

    # Load the data path and changelog path
    data_path = file_paths['data_path'][0]['path']
    changelog_path = file_paths['changelog_path'][0]['path']
    new_labels_path = file_paths['new_labels_path'][0]['path']
    print(new_labels_path)

    # Load the data
    studydata = pd.read_csv(data_path)

    # Load new lables
    new_labels_df = pd.read_csv(new_labels_path)

    # Swap column labels from studydata with new labels
    studydata.columns = new_labels_df['new_labels']


    # Get one hot encoding for specific columns
    one_hot_encoded_df = one_hot_encoding(studydata, columns=['work_remotely','mh_diagnosis', 'mh_professional_diagnosis', 'country', 'us_state', 'country_work', 'us_state_work', 'work_position'])
    save_dataframe_to_csv(one_hot_encoded_df, data_path, changelog_path, 'one_hot_encoded_df')


   
    print("all donekkk6")