import os
import shutil

import numpy as np
import pandas as pd

from tqdm import tqdm

from src.utils import load_json


def add_negative_samples(df, item_set, neg_to_pos_ratio):
    user_positive_examples = df[["uid", "item"]].groupby("uid").agg(list).reset_index().values.tolist()
    user_positive_examples = {k[0]: k[1] for k in user_positive_examples}
    for user_id in tqdm(user_positive_examples.keys()):
        num_neg_samples = int(np.ceil(neg_to_pos_ratio * len(user_positive_examples[user_id])))
        neg_candidates = list(item_set - set(user_positive_examples[user_id]))
        neg_items = np.random.choice(neg_candidates, num_neg_samples, replace=False)
        neg_examples = [{"uid": user_id, "item": j, "label": 0} for j in neg_items]
        df = df.append(neg_examples, ignore_index=True)
    return df


def normalize(type, *args, **kwargs):
    def binary(v):
        if v > 0: return 1
        return 0

    def linear(v):
        return kwargs["a"] * v + kwargs["b"]
    
    def log_linear(v):
        return kwargs["a"] * np.log10(v) + kwargs["b"]
    
    d = {
        "binary": binary,
        "linear": linear,
        "log_linear": log_linear
    }
    return d[type]


def main(config_path, force):
    config = load_json(config_path)
    data_config = config["data"]
    raw_data_dir = data_config["path"]["raw"]
    processed_data_dir = data_config["path"]["processed"]
    if os.path.exists(processed_data_dir):
        if force:
            shutil.rmtree(processed_data_dir)
        else:
            raise ValueError(f"{processed_data_dir} already exists!")
    os.makedirs(processed_data_dir, exist_ok=True)

    shutil.copy(
        os.path.join(raw_data_dir, "metadata.csv"),
        os.path.join(processed_data_dir, "metadata.csv"))

    train_file = os.path.join(raw_data_dir, "train.csv")
    train_df = pd.read_csv(train_file)

    item_map_file = os.path.join(raw_data_dir, "item_map.csv")
    item_map_df = pd.read_csv(item_map_file)
    item_set = set(item_map_df["item"].tolist())

    if "normalize" in data_config:
        norm_func = normalize(**data_config["normalize"])
        train_df = train_df["label"].map(lambda v: norm_func(v))
    train_df = add_negative_samples(train_df, item_set, data_config["neg_to_pos_ratio"])
    train_df = train_df.sample(frac=1, random_state=442).reset_index(drop=True)
    train_df.to_csv(os.path.join(processed_data_dir, "train.csv"), index=False)
