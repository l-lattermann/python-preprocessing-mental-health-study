import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np


data_path = r'C:\Users\Laurenz\Documents\00AA UNI\3. Semester\Machine Learning - Unsupervised Learning and Feature Engineering\Project\Data\mental-heath-in-tech-2016_20161114.csv'

# Shorten column labels
def shorten_column_lables(dataframe, column_name_abbreviations):
    """
    Shortens the column labels of a DataFrame using a list of abbreviations.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame whose column labels should be shortened.
    column_name_abbreviations : list
        A list of abbreviations to be used as new column labels.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with shortened column labels.
    """

    # Create a dictionary with column indices as keys and column names as values
    column_index_mapping = {column: index for index, column in enumerate(dataframe.columns)}

    # Merge abbreviations with column names dictionary
    if len(column_index_mapping) == len(column_name_abbreviations):
        
        # Merge abbreviations with column names dictionary
        for key in column_index_mapping:
            column_index_mapping[key] = column_name_abbreviations[column_index_mapping[key]]
        
    else:
        print("The two lists have different lengths.")

    # Apply new column indices to the DataFrame
    dataframe.rename(columns=column_index_mapping, inplace=True)

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

# Write DataFrame to CSV in readable format
def df_to_readable_csv(dataframe, path: str):
    """
    Writes a DataFrame to a CSV file in a readable format.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame to be written to a CSV file.
    path : str
        The path where the CSV file should be saved.

    Returns
    -------
    None
    """
    # Write the DataFrame to a CSV file
    dataframe.to_csv(path, index=False, line_terminator='\n')

    return None

# Save DataFrame to CSV
def save_dataframe_to_csv(dataframe, name: str):
    """
    Saves a DataFrame to a CSV file.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame to be saved.
    path : str
        The path where the CSV file should be saved.

    Returns
    -------
    None
    """
    
    # Write the DataFrame to a CSV file
    dataframe.to_csv(r'..\..\..\Data\{}.csv'.format(name), index=False)

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
        




    








if __name__ == "__main__":
# Use the specified path for the CSV file
    studydata = pd.read_csv(data_path)

    # Create a dictionary with abbreviated column names
    column_name_abbreviations = ['self_emp',
                    'company_size',
                    'tech_employer',
                    'tech_role',
                    'mental_health_benefits',
                    'aware_mental_health_options',
                    'discussed_mental_health',
                    'mental_health_resources',
                    'anonymity_mental_health',
                    'mental_health_leave_comfort',
                    'neg_mental_health_conseq_employer',
                    'neg_physical_health_conseq_employer',
                    'comfortable_with_coworkers',
                    'comfortable_with_supervisor',
                    'employer_takes_mental_health_seriously',
                    'neg_consequences_mental_health',
                    'medical_coverage_mental_health',
                    'aware_local_resources',
                    'reveal_to_clients',
                    'neg_conseq_revealed_client',
                    'reveal_to_coworkers',
                    'neg_conseq_revealed_coworker',
                    'mental_health_affects_productivity',
                    'work_time_affected_by_mental_health',
                    'previous_employers',
                    'previous_employers_mental_health_benefits',
                    'aware_prev_mental_health_options',
                    'prev_discussed_mental_health',
                    'prev_provided_resources',
                    'prev_anonymity_protected',
                    'prev_neg_mental_health_conseq',
                    'prev_neg_physical_health_conseq',
                    'discuss_with_prev_coworkers',
                    'discuss_with_prev_supervisor',
                    'prev_employers_takes_mental_health_seriously',
                    'prev_negative_consequences',
                    'physical_health_with_potential_employer',
                    'why_not_physical_health',
                    'mental_health_with_potential_employer',
                    'why_not_mental_health',
                    'mental_health_hurt_career',
                    'team_view_negatively',
                    'share_with_family',
                    'observed_bad_handling',
                    'observed_bad_handling_effect',
                    'family_history_mental_health',
                    'past_mental_health_disorder',
                    'current_mental_health_disorder',
                    'diagnosed_conditions',
                    'suspected_conditions',
                    'diagnosed_by_professional',
                    'conditions_diagnosed_by_professional',
                    'sought_mental_health_treatment',
                    'mental_health_treated_interferes',
                    'mental_health_not_treated_interferes',
                    'age', 'gender',
                    'country_live',
                    'state_live',
                    'country_work',
                    'state_work',
                    'work_position',
                    'remote_work']

    # Display the first few rows of the DataFrame
    print(studydata.head())

    shorten_column_lables(studydata, column_name_abbreviations)

    print(studydata.head())

    # Extract all unique features and their values
    unique_feature_df = pd.DataFrame()
    unique_feature_df = extract_all_unique_feature_values(studydata)
    print(unique_feature_df.head())


    # Write the DataFrame to a CSV file
    save_dataframe_to_csv(unique_feature_df, 'shortened_data')

    # Get the value counts for each feature
    value_counts_df = get_value_count_per_feature(unique_feature_df)
    print(value_counts_df.to_string())

    # Extract features with extraordinary value counts
    extraordinary_value_counts = extract_features_extraodinary_value_counts(value_counts_df, threshold_over_mean = 0.3)

    # Display the features with extraordinary value counts as a bar plot in matplotlib
    """ extraordinary_value_counts.plot(kind='bar', figsize=(6, 6))
    plt.tight_layout(pad=1.0)
    plt.show() """

    # Impute missing values in the DataFrame
    impute_missing_numeric_values(studydata)

    print("all donekkk6")