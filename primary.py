"""
Descripttion: DeepModelIPProtection
version: 1.0
Author: XtHhua
Date: 2024-04-08 23:29:37
LastEditors: XtHhua
LastEditTime: 2024-04-08 23:33:25
"""

from functools import partial

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.utils.data import Subset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data._utils import collate
from torchvision.datasets import CIFAR10
from torchvision import transforms

from utils import timer
from utils import save_result
from model_loader import ModelLoader


class PrimaryFingerprint:
    def __init__(self, model: Module, trainset: Dataset, domain: str) -> None:
        self.model = model
        self.trainset = trainset
        self.domain = domain
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @timer
    def generate_primary_fingerprint(self, n: int) -> None:
        dataloader = DataLoader(dataset=self.trainset, shuffle=False, batch_size=1000)
        correct_info, wrong_info = self.test_pro(
            model=self.model, dataloader=dataloader
        )
        correct_partial = partial(self.confidence_well, info=correct_info, n=n)
        correct_partial = partial(self.confidence_well, info=correct_info, n=n)
        for m in ["cc", "cu"]:
            correct_partial(component=m)
        wrong_partial = partial(self.confidence_well, info=wrong_info, n=n)
        for m in ["wc", "wu"]:
            wrong_partial(component=m)

    def confidence_well(self, info: list, component: str, n: int) -> None:
        if component in ["cc", "wu"]:
            reverse = False
        elif component in ["wc", "cu"]:
            reverse = True
        else:
            raise NotImplementedError
        n_loss_indexs = sorted(info, key=lambda x: x[0], reverse=reverse)[:n]
        _, indexs = zip(*n_loss_indexs)
        sub_dataset = Subset(self.trainset, indexs)
        data, label = [], []
        for item in sub_dataset:
            data.append(item[0])
            label.append(item[1])
        data = torch.stack(data, dim=0)
        label = torch.tensor(label)

        save_result(
            path=f"./fingerprint/{self.domain}/primary_{component}_original.pkl",
            data={"data": data, "label": label},
        )

    def test_pro(self, dataloader: DataLoader):
        self.model.eval()
        model = self.model.to(self.device)
        correct_num = 0
        correct, wrong = [], []
        for _, batch_index in enumerate(dataloader._index_sampler):
            batch_data = collate.default_collate(
                [dataloader.dataset[i] for i in batch_index]
            )
            b_x = batch_data[0].to(self.device)
            b_y = batch_data[1].to(self.device)
            output = model(b_x)
            loss = F.cross_entropy(output, b_y, reduction="none")
            pred = torch.argmax(output, dim=-1)
            correct.extend(
                [
                    (loss[i].detach().cpu(), batch_index[i])
                    for i, label in enumerate(pred)
                    if label == b_y[i]
                ]
            )
            wrong.extend(
                [
                    (loss[i].detach().cpu(), batch_index[i])
                    for i, label in enumerate(pred)
                    if label != b_y[i]
                ]
            )
            correct_num += (pred == b_y).sum().item()
        model.cpu()
        assert correct_num == len(correct)
        return correct, wrong


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = ModelLoader.domain_ada("CIFAR10").load_model(index=0, model_type="source")

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    trainset = CIFAR10(
        root="./data", train=True, download=True, transform=transform_test
    )
    pf = PrimaryFingerprint(model=source, trainset=trainset, domain="CIFAR10")
    pf.generate_primary_fingerprint(n=80)
