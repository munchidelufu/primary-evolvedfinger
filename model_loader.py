import os

import torch
from torch.nn import Module
from torchvision import models

from models_arc import Bert
from models_arc import DPCNN
from models_arc import TextCNN
from models_arc import TextRCNN
from models_arc import TextRNN
from models_arc import Conformer
from models_arc import DeepConvNet
from models_arc import ShallowConvNet


class ModelLoader:
    @staticmethod
    def domain_ada(domain: str):
        if domain in ["CIFAR10", "CIFAR100"]:
            return CVModelLoader(domain=domain)
        elif domain in ["THUCNews", "OnlineShop"]:
            return NLPModelLoader(domain=domain)
        elif domain in ["DEAP", "MAHNOBHCI"]:
            return BCIModelLoader(domain=domain)
        else:
            raise NotImplementedError()


class DomainModelLoader:
    def load_model(self, mode: str, index: int = 0):
        raise NotImplementedError()


class CVModelLoader(DomainModelLoader):
    def __init__(self, domain: str):
        self.domain = domain
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_dir = f"./model_weights/{self.domain}"
        if self.domain == "CIFAR100":
            self.classes = 100
        elif self.domain == "CIFAR10":
            self.classes = 10
        else:
            raise NotImplementedError

    def load_model(self, mode: str, index: int = 0) -> Module:
        if mode == "source":
            model = models.vgg16_bn(weights=None)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, self.classes)
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        self.base_dir,
                        f"{mode}",
                        "model_best.pth",
                    ),
                    self.device,
                )
            )
        elif mode in ["surrogate", "ft", "tl"]:
            model = models.vgg16_bn(weights=None)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, self.classes)
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        self.base_dir,
                        f"{mode}",
                        f"model_best_{index}.pth",
                    ),
                    self.device,
                )
            )
        elif mode in ["fp"]:
            model = torch.load(
                os.path.join(
                    self.base_dir,
                    f"{mode}",
                    f"model_best_{index}.pth",
                ),
                self.device,
            )
        elif mode in [
            "ml",
            "mp",
            "ma",
        ]:
            if index < 5:
                model = models.vgg13(weights=None)
                in_feature = model.classifier[-1].in_features
                model.classifier[-1] = torch.nn.Linear(in_feature, self.classes)
            elif 5 <= index < 10:
                model = models.resnet18(weights=None)
                in_feature = model.fc.in_features
                model.fc = torch.nn.Linear(in_feature, self.classes)
            elif 10 <= index < 15:
                model = models.densenet121(weights=None)
                in_feature = model.classifier.in_features
                model.classifier = torch.nn.Linear(in_feature, self.classes)
            else:
                model = models.mobilenet_v2(weights=None)
                in_feature = model.classifier[-1].in_features
                model.classifier[-1] = torch.nn.Linear(in_feature, self.classes)
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        self.base_dir,
                        f"{mode}",
                        f"model_best_{index}.pth",
                    ),
                    self.device,
                )
            )
        elif mode in ["irrelevant"]:
            if 0 <= index < 5:
                model = models.vgg13(weights=None)
                in_feature = model.classifier[-1].in_features
                model.classifier[-1] = torch.nn.Linear(in_feature, self.classes)
            elif 5 <= index < 10:
                model = models.resnet18(weights=None)
                in_feature = model.fc.in_features
                model.fc = torch.nn.Linear(in_feature, self.classes)
            elif 10 <= index < 15:
                model = models.densenet121(weights=None)
                in_feature = model.classifier.in_features
                model.classifier = torch.nn.Linear(in_feature, self.classes)
            elif 15 <= index < 20:
                model = models.mobilenet_v2(weights=None)
                in_feature = model.classifier[-1].in_features
                model.classifier[-1] = torch.nn.Linear(in_feature, self.classes)
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        self.base_dir,
                        f"{mode}",
                        f"model_best_{index}.pth",
                    ),
                    self.device,
                )
            )
        return model


class NLPModelLoader(DomainModelLoader):
    def __init__(self, domain: str):
        self.domain = domain
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_dir = f"./model/{self.domain}"
        if self.domain == "THUCNews":
            self.classes = 10
        elif self.domain == "OnlineShop":
            self.classes = 10
        else:
            raise NotImplementedError

    def load_model(self, mode: str, index: int = 0):
        if mode == "source":
            model = Bert()
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        self.base_dir,
                        f"{mode}",
                        "model_best.pth",
                    ),
                    self.device,
                )
            )
        elif mode in ["surrogate", "ft", "tl"]:
            model = Bert()
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        self.base_dir,
                        f"{mode}",
                        f"model_best_{index}.pth",
                    ),
                    self.device,
                )
            )
        elif mode in ["fp"]:
            model = torch.load(
                os.path.join(
                    self.base_dir,
                    f"{mode}",
                    f"model_best_{index}.pth",
                ),
                self.device,
            )
        elif mode in [
            "ml",
            "mp",
            "irrelevant",
        ]:
            if index < 5:
                model = DPCNN()
            elif 5 <= index < 10:
                model = TextCNN()
            elif 10 <= index < 15:
                model = TextRCNN()
            else:
                model = TextRNN()
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        self.base_dir,
                        f"{mode}",
                        f"model_best_{index}.pth",
                    ),
                    self.device,
                )
            )
        return model


class BCIModelLoader(DomainModelLoader):
    def __init__(self, domain: str) -> None:
        self.domain = domain
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_dir = f"./model/{self.domain}"
        if self.domain == "DEAP":
            self.classes = 2
        elif self.domain == "MAHNOBHCI":
            self.classes = 2
        else:
            raise NotImplementedError

    def load_model(self, mode: str, index: int = 0):
        if mode == "source":
            model = Conformer(n_classes=2)
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        self.base_dir,
                        f"{mode}",
                        f"model_best.pth",
                    ),
                    self.device,
                )
            )
        elif mode in ["surrogate", "ft", "tl"]:
            model = Conformer(n_classes=2)
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        self.base_dir,
                        f"{mode}",
                        f"model_best_{index}.pth",
                    ),
                    self.device,
                )
            )
        elif mode in ["fp"]:
            model = torch.load(
                os.path.join(
                    self.base_dir,
                    f"{mode}",
                    f"model_best_{index}.pth",
                ),
                self.device,
            )
        elif mode in [
            "ml",
            "mp",
            "ma",
            "irrelevant",
        ]:
            if index < 5:
                model = Conformer(n_classes=2)
            elif 5 <= index < 10:
                model = DeepConvNet(n_classes=2)
            elif 10 <= index < 15:
                model = ShallowConvNet(n_classes=2)
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        self.base_dir,
                        f"{mode}",
                        f"model_best_{index}.pth",
                    ),
                    self.device,
                )
            )
        return model
