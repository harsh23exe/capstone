import json
import os
import ast

def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


root = get_project_root()
config_dir = os.path.join(root, "utils")
config_path = os.path.join(config_dir, "configs.json")


with open(config_path, "r") as file:
    config = json.load(file)

ptm = config["ptm"]
emb_dir = config["emb_dir"]
model_name = config["model_name"]
dataset_dir = config["dataset_dir"]