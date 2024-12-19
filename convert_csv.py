import pandas as pd 
import json

paths = 'file_paths.json'

file_paths = json.load(open(paths, 'r', encoding='latin1')) # Load the file paths JSON

changes_path = file_paths['all_changes_path'][0]['path']    # Get the data path
changes = pd.read_csv(changes_path, delimiter=";", encoding='utf-8')       # Load the changes data


changes.to_csv('data/all_changes2.csv', index=False) # Save the changes data to a CSV file