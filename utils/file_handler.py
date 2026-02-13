import json
import pandas as pd

def read_json(path):
    """Read a JSON file and return a Python object (dict or list)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data, path, indent=2):
    """Write a Python object (dict or list) to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def read_csv(path):
    """Read a CSV file and return a pandas DataFrame."""
    return pd.read_csv(path)


def write_csv(data, path, index=False):
    """
    Write a pandas DataFrame to a CSV file.
    
    data: pandas DataFrame
    index: whether to save index column
    """
    data.to_csv(path, index=index)
