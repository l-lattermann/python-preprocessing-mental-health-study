import json
from pathlib import Path
import inspect
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
import sklearn as sk


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
    changes = pd.read_csv(all_changes_path, encoding='utf-8')

    # Replace all non ascii characters in studydata
    matches_columns = study_data.columns.str.extractall(r'([\xa0]+)')# Iterate over all string columns and extract non-ASCII characters
    print("_"*100)
    print(f"Matching columns: {matches_columns}")
    print("_"*100)

    # Replace Â  with a regular space
    print("_"*100)
    study_data.columns.str.replace('\xa0', ' ', regex= False)
    study_data.columns = study_data.columns.str.replace(r'([\xa0]+)', '', regex=True)
    matching_columns = study_data.columns[study_data.columns.str.contains('you have medical coverage', case=False, na=False)]
    if matching_columns.empty:
        print("No matching columns found")
    elif not matching_columns.empty:
        matching_columns = matching_columns[0]
    print(f"matching columns are: {matching_columns}")
    print("_"*100)