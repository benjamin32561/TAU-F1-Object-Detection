import os
import random

def ListFilesInFolder(path):
    """
    List all files in the specified directory with their full paths.

    Args:
        path (str): The path to the directory to list files in.

    Returns:
        A list of tuples containing the filename and its full path in the directory.
    """
    # Get a list of all files in the directory
    files = os.listdir(path)

    # Filter out directories
    files = [os.path.join(path, file) for file in files if os.path.isfile(os.path.join(path, file))]

    return files

def GetRandomFromList(arr:list):
    return arr[random.randint(0, len(arr)-1)]
