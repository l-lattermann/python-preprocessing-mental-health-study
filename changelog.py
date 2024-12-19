import json
import inspect
import pandas as pd
import numpy as np
from tabulate import tabulate

_sentinel = object()  # Sentinel for detecting "no content"

class Changelog:
    """
    A class to write to a changelog file.

    Attributes
    ----------
    path : str
        The path to the changelog file.
    
    Methods
    -------
    write(message: str, content: any = _sentinel)
        Writes content to a changelog file.
    
    append(message: str, content: any = _sentinel)
        Appends content to the last message in a changelog file.
    
    
    """

    def __init__(self, JSON_path: bool = False) -> None: 
        """
        Initializes a changelog file in 'w' mode. A header is written to the file.

        Parameters
        ----------
        changelog_path : str
            The path to the changelog file.
        """
        # Load file paths from json file
        if JSON_path:
            try:
                with open('file_paths.json', 'r', encoding='latin1') as paths:
                    file_paths = json.load(paths)
                    changelog_path = file_paths['changelog_path'][0]['path']
                    self.path = changelog_path
            except FileNotFoundError:
                self.path = "changelog.txt"

        # Open the changelog file in write mode and write the header
        with open(self.path, 'w', encoding='utf-8') as changelog:

            # Write the header
            changelog.write("="*200 + "\n"+ "="*200 +"\n")
            changelog.write(pd.Timestamp.now().strftime("%Y.%m.%d. %H:%M:1%S ") + "Changelog initialized.\n")
            changelog.write("="*200 + "\n" + "="*200 + "\n")

    def write(self, message: str, content: any = _sentinel) -> None:
        """
        Writes content to a changelog file.

        Parameters
        ----------
        message : str
            The message to be written to the changelog file.
        content : any
            The content to be written to the changelog file.
        """
        # Get the functionname that called write_to_changelog
        calling_function = inspect.stack()[1].function

        # Open the changelog file in append mode and write the header
        with open (self.path, 'a', encoding='utf-8') as changelog: 
            changelog.write("\n"+ "_"*200 + "\n")
            changelog.write(pd.Timestamp.now().strftime("%Y.%m.%d. %H:%M:1%S ") + "within  " + calling_function + ":\n")
            changelog.write(message)

            # Return if no content is provided
            if content is _sentinel:
                return
            
            # Check if content is a DataFrame
            if isinstance(content, pd.DataFrame):
                # Check if contetnt is empty
                if content.empty:
                    changelog.write("\n None")
                    return
                # Write the dataframe to the changelog file
                changelog.write("\n" + tabulate(content, headers='keys', tablefmt='simple'))
                changelog.write("\n\n" + "Content datatype: " + str(type(content)))

            # Check if content is a index
            elif isinstance(content, pd.Index):
                # Check if content is empty
                if content.empty:
                    changelog.write("\n None")
                    return
                # Write the index to the changelog file
                changelog.write("\n" + tabulate(content.to_frame(), headers='keys', tablefmt='simple'))
                changelog.write("\n\n" + "Content datatype: " + str(type(content)))

            # Check if content is a dictionary
            elif isinstance(content, dict):
                # Check if content is empty
                if not content:
                    changelog.write("\n None")
                    return
                # Write the dictionary to the changelog file
                for key, value in content.items():
                    changelog.write(f"\n{key}: {value}")
                changelog.write("\n\n" + "Content datatype: " + str(type(content)))

            # Check if content is a list or tuple
            elif isinstance(content, (list, tuple)):
                # Check if content is empty
                if not content:
                    changelog.write("\n None")
                    return
                # Write the list or tuple to the changelog file
                changelog.write("\n" + '\n'.join(map(str, content)))
                changelog.write("\n\n" + "Content datatype: " + str(type(content)))

            # Check if content is a array
            elif isinstance(content, np.ndarray):
                # Check if content is empty
                if not content.size:
                    changelog.write("\n None")
                    return
                # Write the array to the changelog file
                changelog.write("\n" + tabulate(pd.DataFrame(content), headers='keys', tablefmt='simple'))
                changelog.write("\n\n" + "Content datatype: " + str(type(content)))

            # Handle general objects by converting to string if possible
            elif isinstance(content, object):
                # Check if content is empty
                if not content:
                    changelog.write("\n None")
                    return
                # Write the object to the changelog file
                if hasattr(content, 'to_string'):
                    changelog.write("\n" + str(content.to_string()))
                    changelog.write("\n\n" + "Content datatype: " + str(type(content)))
                else:
                    changelog.write("\n" + str(content))
                    changelog.write("\n\n" + "Content datatype: " + str(type(content)))
            else:
                # Fallback for primitives (int, float, etc.)
                try:
                    changelog.write("\n" + str(content))
                    changelog.write("\n\n" + "Content datatype: " + str(type(content)))
                except TypeError:
                    changelog.write("\n There was an error writing the content to the changelog file.")
                    changelog.write("\n\n" + "Content datatype: " + str(type(content)))


    # Function to append to last message in changelog
    def append(self, message: str, content: any = _sentinel) -> None:
        """
        Appends content to the last message in a changelog file.

        Parameters
        ----------
        message : str
            The message to be appended to the changelog file.
        content : any
            The content to be appended to the changelog file.
        """
        # Get the functionname that called write_to_changelog
        calling_function = inspect.stack()[1].function

        # Open the changelog file in append mode and write the header
        with open (self.path, 'a', encoding='utf-8') as changelog: 
            changelog.write(message)

            # Return if no content is provided
            if content is _sentinel:
                return
            
            # Check if content is a DataFrame
            if isinstance(content, pd.DataFrame):
                # Check if contetnt is empty
                if content.empty:
                    changelog.write("\n None")
                    return
                # Write the dataframe to the changelog file
                changelog.write("\n" + tabulate(content, headers='keys', tablefmt='simple'))
                changelog.write("\n\n" + "Content datatype: " + str(type(content)))

            # Check if content is a index
            elif isinstance(content, pd.Index):
                # Check if content is empty
                if content.empty:
                    changelog.write("\n None")
                    return
                # Write the index to the changelog file
                changelog.write("\n" + tabulate(content.to_frame(), headers='keys', tablefmt='simple'))
                changelog.write("\n\n" + "Content datatype: " + str(type(content)))

            # Check if content is a dictionary
            elif isinstance(content, dict):
                # Check if content is empty
                if not content:
                    changelog.write("\n None")
                    return
                # Write the dictionary to the changelog file
                for key, value in content.items():
                    changelog.write(f"\n{key}: {value}")
                changelog.write("\n\n" + "Content datatype: " + str(type(content)))

            # Check if content is a list or tuple
            elif isinstance(content, (list, tuple)):
                # Check if content is empty
                if not content:
                    changelog.write("\n None")
                    return
                # Write the list or tuple to the changelog file
                changelog.write("\n" + '\n'.join(map(str, content)))
                changelog.write("\n\n" + "Content datatype: " + str(type(content)))

            # Check if content is a array
            elif isinstance(content, np.ndarray):
                # Check if content is empty
                if not content.size:
                    changelog.write("\n None")
                    return
                # Write the array to the changelog file
                changelog.write("\n" + tabulate(pd.DataFrame(content), headers='keys', tablefmt='simple'))
                changelog.write("\n\n" + "Content datatype: " + str(type(content)))

            # Handle general objects by converting to string if possible
            elif isinstance(content, object):
                # Check if content is empty
                if not content:
                    changelog.write("\n None")
                    return
                # Write the object to the changelog file
                if hasattr(content, 'to_string'):
                    changelog.write("\n" + str(content.to_string()))
                    changelog.write("\n\n" + "Content datatype: " + str(type(content)))
                else:
                    changelog.write("\n" + str(content))
                    changelog.write("\n\n" + "Content datatype: " + str(type(content)))
            else:
                # Fallback for primitives (int, float, etc.)
                try:
                    changelog.write("\n" + str(content))
                    changelog.write("\n\n" + "Content datatype: " + str(type(content)))
                except TypeError:
                    changelog.write("\n There was an error writing the content to the changelog file.")
                    changelog.write("\n\n" + "Content datatype: " + str(type(content)))
        

    def write_highlighted (self, message: str, content: any = _sentinel) -> None:
        with open (self.path, 'a', encoding='utf-8') as file:
            file.write("\n" + "+"*2000 + "\n")   # Write a line of "+" characters to the changelog file
        
        self.write(message, content)            # Write the message to the changelog file

        with open (self.path, 'a', encoding='utf-8') as file:
            file.write("\n" + "+"*2000 + "\n")   # Write a line of "+" characters to the changelog file