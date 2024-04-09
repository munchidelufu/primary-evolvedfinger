import utils
import copy
from torcheeg import transforms
from torcheeg.datasets.constants.emotion_recognition import (
    DEAP_CHANNEL_LOCATION_DICT,
    MAHNOB_CHANNEL_LOCATION_DICT,
)
import sys
from torcheeg.datasets import DEAPDataset, MAHNOBDataset
from multiprocessing import Pool


def load_deap_dataset(model_name: str, label: str = "valence"):
    root_path = "./data/deap/normalize"
    io_path = f"./data/deap/{model_name}"
    dataset = DEAPDataset(
        io_path=io_path,
        root_path=root_path,
        offline_transform=transforms.MeanStdNormalize(axis=1),
        online_transform=transforms.Compose([transforms.ToTensor(), transforms.To2d()]),
        label_transform=transforms.Compose(
            [
                transforms.Select(label),
                transforms.Binary(5.0),
            ]
        ),
        num_worker=4,
    )
    return dataset


def load_mahnob_dataset(model_name: str, label: str = "feltVlnc"):
    root_path = "/data/dataset/Mahnob_HCI_tagging/Sessions/"
    io_path = f"./data/mahnobhci/{model_name}"
    dataset = MAHNOBDataset(
        io_path=io_path,
        root_path=root_path,
        offline_transform=transforms.MeanStdNormalize(axis=1),
        online_transform=transforms.Compose([transforms.ToTensor(), transforms.To2d()]),
        label_transform=transforms.Compose(
            [
                transforms.Select(label),
                transforms.Binary(5.0),
            ]
        ),
        num_worker=4,
    )
    return dataset


def load_split_dataset(model_name="ccnn", label="valence"):
    dataset = load_deap_dataset(model_name=model_name, label=label)
    train_info, attack_info = utils.split_dataset_by_trial(dataset=dataset)

    train_dataset = copy.deepcopy(dataset)
    train_dataset.info = train_info

    attack_dataset = copy.deepcopy(dataset)
    attack_dataset.info = attack_info
    return train_dataset, attack_dataset


def is_balance(dataset: DEAPDataset):
    one = 0
    zero = 0
    for i in range(len(dataset)):
        if dataset[i][1] == 1:
            one += 1
        elif dataset[i][1] == 0:
            zero += 1
    print(f"high_valance:{one}, low_valance:{zero}")
