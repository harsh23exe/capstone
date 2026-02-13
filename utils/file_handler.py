import json

def read_json(path):
    """Read a JSON file and return a Python object (dict or list)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data, path, indent=2):
    """Write a Python object (dict or list) to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
