import os
import csv
from tqdm import tqdm
from functools import partial

import torch
import numpy as np

from model_loader import ModelLoader
from primary import PrimaryFingerprint
from evolved import EvolvedFingerprint
from detect_erase_attack import QueryAttack
from detect_erase_attack import SynonymAttack
from detect_erase_attack import InputSmooth
from utils import load_result
from utils import calculate_auc


class ModelFingerMatch:
    def __init__(
        self,
        n: int,
        priorevo: str = "primary",
        domain: str = "CIFAR10",
        detection_attack: bool = False,
        model_type: list = None,
        model2num: dict = None,
    ) -> None:
        self.n = n
        if priorevo not in ["primary", "evolved"]:
            raise ValueError("priorevo must be primary or evolved")
        self.finger = priorevo
        self.domain = domain
        self.model_loader = ModelLoader.domain_ada(self.domain)
        self.finger_components = ["cc", "wc", "cu", "wu"]
        self.model_type = model_type
        self.model2num = model2num
        self.detection_attack = detection_attack
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if detection_attack:
            if domain in ["CIFAR10", "CIFAR100"]:
                self.attack = QueryAttack()
            elif domain in ["THUCNews"]:
                self.attack = SynonymAttack()
            elif domain in ["DEAP"]:
                self.attack = InputSmooth()
            else:
                raise NotImplementedError

    def model_feature(self):
        model_features_fix = partial(
            self.model_feature_helper,
            *self.finger_components,
            self.n,
        )
        for mt in self.model_type:
            model_features_fix(model_type=mt)

    def model_feature_helper(self, *finger_components, model_type: str = "source"):
        tqdm_bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        tqdm_kwargs = {
            "desc": model_type,
            "bar_format": tqdm_bar_format,
            "ncols": 80,
            "bar_format": "\033[32m" + tqdm_bar_format + "\033[0m",
            "leave": False,
        }
        bar = tqdm(range(self.model2num[model_type]), **tqdm_kwargs)
        save_dir = (
            f"./model_feature/{self.domain}/"
            if self.detection_attack
            else f"./model_feature/{self.domain}"
        )
        os.makedirs(save_dir, exist_ok=True)
        feature_file = (
            f"{self.finger}_original.csv"
            if self.detection_attack
            else f"{self.finger}_attack.csv"
        )
        with open(os.path.join(save_dir, feature_file), mode="a", newline="") as file:
            writer = csv.writer(file)
            for i in range(self.model2num[model_type]):
                feature_record = []
                model = self.model_loader.load_model(index=i, mode=model_type)
                model.eval()
                for fc in finger_components:
                    path = (
                        f"./fingerprint/{self.domain}/{self.finger}_{fc}_original.pkl"
                        if self.detection_attack
                        else f"./fingerprint/{self.domain}/{self.finger}_{fc}_attack.pkl"
                    )
                    dataset = load_result(path=path)
                    data = dataset["data"].to(self.device)
                    label = dataset["label"].to(self.device)
                    pred = torch.argmax(model(data.to(self.device)), dim=1)
                    correct = (label == pred).sum().item()
                    feature_record.append(round(correct / self.n, 2))
                feature_record.append(model_type)
                bar.update(1)
                writer.writerow(feature_record)

    def model_similarity(
        self,
        model_feature_path: str,
        features: list = [0, 1, 2, 3],
        verbose: bool = False,
    ):
        with open(model_feature_path, mode="r") as file:
            reader = csv.reader(file)
            features = [[float(row[i]) for i in features] + [row[4]] for row in reader]
        #
        source_feature = np.array([row[:-1] for row in features if row[-1] == "source"])
        irr_feature = np.array(
            [row[:-1] for row in features if row[-1] == "irrelevant"]
        )
        pro_feature = np.array([row[:-1] for row in features if row[-1] == "mp"])
        lab_feature = np.array([row[:-1] for row in features if row[-1] == "ml"])
        tl_feature = np.array([row[:-1] for row in features if row[-1] == "tl"])
        fp_feature = np.array([row[:-1] for row in features if row[-1] == "fp"])
        ft_feature = np.array([row[:-1] for row in features if row[-1] == "ft"])
        adv_feature = np.array([row[:-1] for row in features if row[-1] == "ma"])

        def model_distance(input):
            input = np.array(input)
            simi_score = np.linalg.norm(input - source_feature[0], ord=2)
            return simi_score

        irr_simi = list(map(model_distance, irr_feature))
        pro_simi = list(map(model_distance, pro_feature))
        lab_simi = list(map(model_distance, lab_feature))
        cif_simi = list(map(model_distance, tl_feature))
        fp_simi = list(map(model_distance, fp_feature))
        ft_simi = list(map(model_distance, ft_feature))
        adv_simi = list(map(model_distance, adv_feature))

        pro_auc = calculate_auc(list_a=pro_simi, list_b=irr_simi)
        lab_auc = calculate_auc(list_a=lab_simi, list_b=irr_simi)
        tl_auc = calculate_auc(list_a=cif_simi, list_b=irr_simi)
        fp_auc = calculate_auc(list_a=fp_simi, list_b=irr_simi)
        ft_auc = calculate_auc(list_a=ft_simi, list_b=irr_simi)
        adv_auc = calculate_auc(list_a=adv_simi, list_b=irr_simi)

        if verbose:
            print(
                "ft:",
                ft_auc,
                "fp:",
                fp_auc,
                "lab:",
                lab_auc,
                "pro:",
                pro_auc,
                "adv:",
                adv_auc,
                "tl:",
                tl_auc,
            )
        auc_records = [pro_auc, lab_auc, tl_auc, fp_auc, ft_auc, adv_auc]
        return sum(auc_records) / len(auc_records)


if __name__ == "__main__":
    mfm = ModelFingerMatch(
        n=80,
        priorevo="primary",
        domain="CIFAR10",
        detection_attack=False,
        model_type=["teacher", "ml", "mp", "irrelevant", "tl", "fp", "ft", "ma"],
        model2num={
            "teacher": 1,
            "ml": 20,
            "mp": 20,
            "irrelevant": 20,
            "tl": 10,
            "fp": 15,
            "ft": 20,
            "ma": 20,
        },
    )
    # inference model feature
    mfm.model_feature()
    #
    mfm.model_similarity(
        model_feature_path="./model_feature/CIFAR10/primary_original.csv"
    )
    mfm.model_similarity(
        model_feature_path="./model_feature/CIFAR10/evolved_original.csv"
    )
