import inspect
import pandas as pd
import numpy as np
from tabulate import tabulate

_sentinel = object()  # Sentinel for detecting "no content"
_changelog_path = "changelog.txt"

def init_changelog(custom_path: str = _changelog_path): 
    """
    Initializes a changelog file in 'w' mode.

    Parameters
    ----------
    changelog_path : str
        The path to the changelog file.
    """
    # Open the changelog file in write mode and write the header
    with open(custom_path, 'w', encoding='utf-8') as changelog:

        # Set custom path for changelog
        global _changelog_path 
        _changelog_path = custom_path

        # Write the header
        changelog.write("="*200 + "\n"+ "="*200 +"\n")
        changelog.write(pd.Timestamp.now().strftime("%Y.%m.%d. %H:%M:1%S ") + "Changelog initialized.\n")
        changelog.write("="*200 + "\n" + "="*200 + "\n")

# Function to write object to changelog
def write(message: str, content: any = _sentinel):
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
    with open (_changelog_path, 'a', encoding='utf-8') as changelog: 
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
def append(message: str, content: any = _sentinel):
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
    with open (_changelog_path, 'a', encoding='utf-8') as changelog: 
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