import pandas as pd
import numpy as np
import json
import sklearn as sk




# Fill missing age values with the median
def fill_missing_values(study_data, changes, log):
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
            changes['encoding'] = changes['encoding'].map(lambda x: json.loads(x) if isinstance(x, str) else x)
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