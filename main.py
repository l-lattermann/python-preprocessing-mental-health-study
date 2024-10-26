import json
from pathlib import Path
import inspect
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
import sklearn as sk
import re


# Function to write object to changelog
def write_to_changelog(path: str, message: str, content: any = None):
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

    with open(path, 'a', encoding='utf-8') as changelog:
        changelog.write("\n\n"+ "_"*200 + "\n")
        changelog.write(pd.Timestamp.now().strftime("%Y.%m.%d. %H:%M:1%S ") + "within" + calling_function + ":\n")
        changelog.write(message + "\n")

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

# Main function
if __name__ == "__main__":
    
    # Load all file paths from json file
    with open('file_paths.json', 'r', encoding='latin1') as file:
        file_paths = json.load(file)

        data_path = file_paths['data_path'][0]['path']
        changelog_path = file_paths['changelog_path'][0]['path']
        all_changes_path = file_paths['all_changes_path'][0]['path']

    
    # Load the study data
    study_data = pd.read_csv(data_path, encoding='utf-8')

    # Write to changelog
    write_to_changelog(changelog_path, "Loaded study data from: " + data_path)
    
    # Load the changes to be made to the study data
    changes = pd.read_csv(all_changes_path, encoding='utf-8')

    # Write to changelog
    write_to_changelog(changelog_path, "Loaded changes data from: " + all_changes_path)

    # Clean weird spaces in the dataset
    matches_columns = study_data.columns.str.extractall(r'([\xa0]+)')                       # Iterate over all string columns and extract non-ASCII characters
    study_data.columns = study_data.columns.str.replace(r'([\xa0]+)', '', regex=True)       # Remove non-breaking spaces
    study_data.columns = study_data.columns.str.strip()                                     # Strip leading and trailing spaces
    for column in study_data.columns:
        replaced_strings = study_data[column].apply(lambda x: str(x).replace('\xa0', ' ') if isinstance(x, str) else x).tolist() # Iterate over all string columns and extract non-ASCII characters
        study_data[column] = study_data[column].apply(lambda x: str(x).replace(r'([\xa0]+)', '') if isinstance( x, str) else x)  # Remove non-breaking spaces
        study_data[column] = study_data[column].apply(lambda x: x.str.strip() if isinstance(x, str) else x)                               # Strip leading and trailing spaces

    # Write to changelog
    write_to_changelog(changelog_path, "Cleaned weird spaces in the dataset. Columns with non-breaking spaces:", matches_columns)
    write_to_changelog(changelog_path, "Cleaned weird spaces in the dataset. Data with non-breaking spaces:", matches_data)

    # Create a copy for later comparison
    study_data_copy = study_data.copy()    

    # Drop rows with over 50% missing values
    original_count = study_data.shape[0]    # Get the original row count
    study_data = study_data.dropna(thresh=study_data.shape[1]/2)
    drop_lines_percent = (1 - (study_data.shape[0] / original_count)) *100    # Calculate the percentage of dropped rows

    # Write to changelog
    write_to_changelog(changelog_path, f"Dropped {drop_lines_percent}% of rows due to missing values.")

    # Apply new shortend column labels
    new_labels_dict = {}    # Dictionary to store the old and new labels
    for old_label in changes['old_labels'].dropna():
        new_labels_dict[old_label] = changes.loc[changes['old_labels'] == old_label, 'new_labels'].values[0]
    study_data.columns = study_data.columns.map(new_labels_dict)

    # Write to changelog
    write_to_changelog(changelog_path, "Applied new shortened column labels as follows:\n", new_labels_dict)

    # Replace age outliers with NaN
    age_copy = study_data['age'].copy()    # Create a copy for later comparison
    study_data['age'] = study_data['age'].apply(lambda x: np.nan if x < 15 or x > 75 else x)    # Age thresholds were manually selected from excel data
    age_checksum = (study_data['age'] != age_copy).sum()    # Check how many rows were changed
    
    # Write to changelog
    write_to_changelog(changelog_path, f"Replaced age outliers with NaN for {age_checksum} rows.")

    # Fill missing age values with the median
    age_copy_2 = study_data['age'].copy()    # Create a copy for later comparison
    age_median = study_data['age'].median()
    study_data.fillna({'age': age_median}, inplace=True)
    age_checksum_2 = (age_copy_2 != study_data['age']).sum()    # Check how many rows were changed

    # Write to changelog
    write_to_changelog(changelog_path, f"Filled missing age values with the median: {age_median}  for {age_checksum_2} rows.")

    # Get the encoding type for the columns
    encoding_types_dict = dict(zip(changes['new_labels'], changes['encoding_type']))

    # Write to changelog
    write_to_changelog(changelog_path, "Encoding types for columns are as follows:", encoding_types_dict)

    # Convert JSON-like strings in 'encoding' column in changes df to actual python dict
    temp_store_encoding = changes['encoding'].copy()    # Create a copy for later comparison
    for encoding_type in changes['encoding_type']:
        if encoding_type in {'bin', 'ord'}:
            changes['encoding'] = changes['encoding'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    # Get non-converted rows
    non_converted_rows = changes.loc[changes['encoding'].apply(lambda x: isinstance(x, str))].index
    # Write to changelog
    bin_ord_checksum = (changes['encoding'] != temp_store_encoding).sum()    # Check how many rows were changed
    write_to_changelog(changelog_path, f"Converted JSON-like strings in 'encoding' column to actual python dict for {bin_ord_checksum} rows.\nRows that were not converted: {non_converted_rows}")

    # Apply binary and ordinal encoding to the respective columns
    bin_ord_applied_list = []   # List to store the columns that binary and ordinal encoding was applied to
    for label, encoding_type in encoding_types_dict.items():
        if encoding_type in {'bin', 'ord'}:

            # Get the encoding dictionary from the changes dataframe
            if isinstance(changes.loc[changes['new_labels'] == label, 'encoding'].values[0], dict):
                encoding = changes.loc[changes['new_labels'] == label, 'encoding'].values[0]

                

                # Map the study data to the encoding
                #print(f"label is: {label} \n","unique values for this label are:\n",study_data[label].unique(), "\nencoding keys for this lable are:\n", encoding.keys(),"\n" + "-"*100)
                study_data[label] = study_data[label].map(encoding)
                

                # Append the label to the list of applied binary and ordinal encodings
                bin_ord_applied_list.append(label)
    
    # Write to changelog
    for label in bin_ord_applied_list:
        write_to_changelog(changelog_path, f"Applied binary and ordinal encoding to the following columns: {label}\n"
                       f"Data looks as follows{study_data[label].dropna().head()}:\n")
        
    
    # Get a list of columns with mostly missing values
    missing_values_columns = study_data.columns[study_data.isna().mean() > 0.99].tolist()

    # Write to changelog
    write_to_changelog(changelog_path, "Columns with mostly missing values:", missing_values_columns)
        

    # Fill missing values with the mean of the column
    columns_fillna_applied_list = []
    columns_fillna_not_applied_list = []
    for column_name in changes['new_labels'].dropna():
        
        encoding_type = changes.loc[changes['new_labels'] == column_name, 'encoding_type'].values[0]
        print("encoding type is: ", encoding_type)

        # Missing values in binary columns get filled with a probabilistic approach to preserve the overall mean
        if changes.loc[changes['new_labels'] == column_name, 'encoding_type'].values[0] == 'bin':

            # Calculate the proportion of 1s and 0s in the binary column
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
                write_to_changelog(changelog_path, "No encoding type found for column: " + column_name)
                # Append the column name to the list of columns not filled missing values
                columns_fillna_not_applied_list.append(column_name)
                continue
        
        # Write to changelog:
        write_to_changelog(changelog_path, "Filled missing values in the following columns: ", columns_fillna_applied_list)
        write_to_changelog(changelog_path, "No fillings were applied in following columns: ", columns_fillna_not_applied_list)
    
    # Apply TF-IDF encoding to the contextual columns
    tfidf_applied_list = []    # List to store the columns that TF-IDF encoding was applied to
    for label, encoding_type in encoding_types_dict:
        if encoding_type == 'TF-IDF':
            # Apply TF-IDF encoding
            tfidf = sk.feature_extraction.text.TfidfVectorizer()
            study_data[label] = tfidf.fit_transform(study_data[label])

            # Append the label to the list of applied TF-IDF encodings
            tfidf_applied_list.append(label)

    # Write to changelog
    write_to_changelog(changelog_path, "Applied TF-IDF encoding to the following columns:", tfidf_applied_list)
    
    # Apply one-hot encoding to the multiple choice columns
    one_hot_applied_list = []    # List to store the columns that one-hot encoding was applied to
    for label, encoding_type in encoding_types_dict:
        if encoding_type == 'one-hot':

            # Initialize the OneHotEncoder
            encoder = sk.preprocessing.OneHotEncoder(sparse_output=False, handle_unknown='ignore')

            # Apply one-hot encoding to the specified column
            one_hot_encoded = encoder.fit_transform(study_data[[label]])

            # Convert to DataFrame and rename columns with the original label prefix
            one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out([label]))

            # Concatenate the new one-hot encoded DataFrame with the original DataFrame
            study_data = pd.concat([study_data.drop(columns=[label]), one_hot_df], axis=1)

            # Append the label to the list of applied one-hot encodings
            one_hot_applied_list.append(label)
    
    # Write to changelog
    write_to_changelog(changelog_path, "Applied one-hot encoding to the following columns:", one_hot_applied_list)

    

            ## fill bin and ord values before one hot and idf encoding. and try to fill one hot and idf missing vlaues during encodign