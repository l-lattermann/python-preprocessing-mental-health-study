import json
import pandas as pd




# Load the study data
def load_data (logger: object) -> pd.DataFrame:
    """
    Loads the study data from JSON path.

    Returns
    -------
    pd.DataFrame
        The DataFrame containing the study data.

    """
    try:
        with open('file_paths.json', 'r', encoding='latin1') as paths:  # Open the file paths JSON
            file_paths = json.load(paths)                               # Load the file paths
            data_path = file_paths['data_path'][0]['path']              # Get the data path

        study_data = pd.read_csv(data_path, encoding='utf-8')           # Load the study data
        logger.write("Loaded study data from: " + data_path)            # Write to changelog
    
    except FileNotFoundError:
        logger.write("Data file not found. Please check the file path in file_paths.json.")

    return study_data

# Load the changes to be made to the study data
def load_changes(logger: object) -> pd.DataFrame:
    """
    Loads the changes to be made to the study data from CSV path.

    Returns
    -------
    pd.DataFrame
        The DataFrame containing the changes to be made to the study data.

    """
    try:
        with open('file_paths.json', 'r', encoding='latin1') as paths:  # Open the file paths JSON
            file_paths = json.load(paths)                               # Load the file paths
            changes_path = file_paths['all_changes_path'][0]['path']    # Get the data path

        changes = pd.read_csv(changes_path, encoding='utf-8')           # Load the changes data
        logger.write("Loaded changes data from: " + changes_path)       # Write to changelog

    except FileNotFoundError:
        logger.write("Changes file not found. Please check the file path in file_paths.json.") 

    return changes

# Save data to csv
def save_to_csv(data: pd.DataFrame, logger: object, file_name: str) -> None:
    """
    Saves the DataFrame to a CSV file.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to be saved.
    file_name : str
        The name of the file to be saved.

    Returns
    -------
    None

    """
    if not file_name:
        file_name = 'cleaned_and_encoded_data.csv'                    # Set the default file name
    if not file_name.endswith('.csv'):                                # Check if the file name ends with .csv
        file_name += '.csv'                                           # Add .csv to the file name
    try:
        data.to_csv(file_name, index=False)                           # Save the DataFrame to a CSV file
        logger.write("Saved data to: " + file_name)                   # Write to changelog

    except FileNotFoundError:
        logger.write("File path not found. Please check the file path in file_paths.json.")
