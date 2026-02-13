import os
from utils.config import *


def create_directory(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)


emb_dir = os.path.join(root, emb_dir)
ptm_emb_dir = os.path.join(emb_dir, ptm)
dataset_dir = os.path.join(root, dataset_dir)

create_directory(ptm_emb_dir)


train_df_path = os.path.join(dataset_dir, f"{ptm}_train.csv")
test_df_path = os.path.join(dataset_dir, f"{ptm}_test.csv")