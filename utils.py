"""
Descripttion: DeepModelIPProtection
version: 1.0
Author: XtHhua
Date: 2024-04-08 23:53:43
LastEditors: XtHhua
LastEditTime: 2024-04-08 23:55:48
"""

import os
import time
import random
import pickle as pkl

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn import model_selection


def seed_everything(seed: int):
    """Set a random seed for reproducibility of results.

    Args:
        seed (int): Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_result(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError
    with open(path, "rb") as file:
        data = pkl.load(file=file)
    return data


def save_result(path: str, data: object):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    #
    with open(path, mode="wb") as file:
        pkl.dump(obj=data, file=file)
        print(f"save to {path} successfully!")


def calculate_auc(list_a, list_b):
    l1, l2 = len(list_a), len(list_b)
    y_true, y_score = [], []
    for i in range(l1):
        y_true.append(0)
    for i in range(l2):
        y_true.append(1)
    y_score.extend(list_a)
    y_score.extend(list_b)
    fpr, tpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=False)
    auc(fpr, tpr)
    return round(auc(fpr, tpr), 2)


def denormalize(batch_data: torch.Tensor):
    min_val = torch.min(batch_data)
    max_val = torch.max(batch_data)
    data = (batch_data - min_val) / (max_val - min_val)
    return data, min_val, max_val


def normalize(batch_data, min_val, max_val):
    data = batch_data * (max_val - min_val) + min_val
    return data


def timer(func: callable) -> callable:
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {(end - start):.2f} seconds to execute.")
        return result

    return wrapper


def split_dataset_by_trial(
    dataset,
    split_ratio=[0.5, 0.5],
    shuffle=True,
):
    split_path = (f"./data/{dataset}/split_dataset_by_trial",)
    if not os.path.exists(split_path):
        os.makedirs(split_path, exist_ok=True)

        info = dataset.info

        trial_ids = list(set(info["trial_id"]))
        train_trial_ids, attack_trial_ids = model_selection.train_test_split(
            trial_ids, test_size=split_ratio[0], random_state=2023, shuffle=shuffle
        )

        train_info = []
        for train_trial_id in train_trial_ids:
            train_info.append(info[info["trial_id"] == train_trial_id])
        train_info = pd.concat(train_info, ignore_index=True)

        attack_info = []
        for attack_trial_id in attack_trial_ids[:-1]:
            attack_info.append(info[info["trial_id"] == attack_trial_id])
        attack_info = pd.concat(attack_info, ignore_index=True)

        train_info.to_csv(os.path.join(split_path, "train.csv"), index=False)
        attack_info.to_csv(os.path.join(split_path, "attack.csv"), index=False)

    train_info = pd.read_csv(os.path.join(split_path, "train.csv"))
    attack_info = pd.read_csv(os.path.join(split_path, "attack.csv"))
    return train_info, attack_info
