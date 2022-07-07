import os
import shutil

import numpy as np
import pandas as pd

from tqdm import tqdm

from src.utils import load_json


def add_negative_samples(df, item_set, neg_to_pos_ratio):
    user_positive_examples = df[["uid", "item"]].groupby("uid").agg(list).reset_index().values.tolist()
    user_positive_examples = {k[0]: k[1] for k in user_positive_examples}
    neg_examples = []
    for i, user_id in tqdm(enumerate(user_positive_examples.keys())):
        num_neg_samples = int(np.ceil(neg_to_pos_ratio * len(user_positive_examples[user_id])))
        neg_candidates = list(item_set - set(user_positive_examples[user_id]))
        if num_neg_samples > len(neg_candidates):
            num_neg_samples = len(neg_candidates)
        neg_items = np.random.choice(neg_candidates, num_neg_samples, replace=False)
        neg_examples.extend([{"uid": user_id, "item": j, "label": 0} for j in neg_items])
        if (i + 1) % 10000 == 0:
            df = df.append(neg_examples, ignore_index=True)
            neg_examples = []
    if len(neg_examples):
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
    raw_data_dir = config["path"]["raw"]
    processed_data_dir = config["path"]["processed"]
    if os.path.exists(processed_data_dir):
        if force:
            shutil.rmtree(processed_data_dir)
        else:
            raise ValueError(f"{processed_data_dir} already exists!")
    os.makedirs(processed_data_dir, exist_ok=True)
    shutil.copyfile(config_path, os.path.join(processed_data_dir, "config.json"))

    resources = ["metadata.csv", "user_map.csv", "item_map.csv", "kw_map.json"]
    for f in resources:
        shutil.copy(
            os.path.join(raw_data_dir, f),
            os.path.join(processed_data_dir, f))

    train_file = os.path.join(raw_data_dir, "train.csv")
    train_df = pd.read_csv(train_file)
    
    val_file = os.path.join(raw_data_dir, "val.csv")
    val_df = pd.read_csv(val_file)

    item_map_file = os.path.join(raw_data_dir, "item_map.csv")
    item_map_df = pd.read_csv(item_map_file)
    item_set = set(item_map_df["index"].tolist())

    if "normalize" in config:
        norm_func = normalize(**config["normalize"])
        train_df = train_df["label"].map(lambda v: norm_func(v))
        val_df = train_df["label"].map(lambda v: norm_func(v))
    train_df = add_negative_samples(train_df, item_set, config["negative_sampling"]["neg_to_pos_ratio"])
    train_df = train_df.sample(frac=1, random_state=442).reset_index(drop=True)
    train_df.to_csv(os.path.join(processed_data_dir, "train.csv"), index=False)

    val_df = add_negative_samples(val_df, item_set, 1)
    val_df.to_csv(os.path.join(processed_data_dir, "val.csv"), index=False)
