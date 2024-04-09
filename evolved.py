from tqdm import tqdm

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Adam
import torch.nn.functional as F

from utils import load_result
from utils import save_result
from utils import timer
from model_loader import ModelLoader


class EvolvedFingerprint:
    def __init__(
        self,
        n: int,
        lr: float,
        tau: int,
        mu: float,
        omega: int,
        model2num: dict,
        domain: str,
    ) -> None:
        self.n = n
        self.lr = lr
        self.tau = tau
        self.mu = mu
        self.omega = omega

        self.finger_components = ["cc", "wc", "cu", "wu"]
        self.model2num = model2num
        self.domain = domain

    def efa(
        self,
        model: Module,
        input: Tensor,
        component: str,
        lr: float,
        tau: int,
    ):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        input = torch.unsqueeze(input, dim=0)
        input_clone = input.clone().detach().requires_grad_(True)
        optimizer = Adam([input_clone], lr=lr)
        p = F.softmax(model(input_clone), dim=1)
        a = torch.argmax(p)
        p_a = p[0][a]
        if component.endswith("c"):
            b = torch.topk(p, k=2, dim=1)[1][:, 1]
            for _ in range(tau):
                p = F.softmax(model(input_clone), dim=1)
                loss = -1 * (p[0][a] - p[0][b]) / (1 - p[0][a] + self.mu)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
        elif component.endswith("u"):
            dis = p[0][a] - 1 / len(p[0])
            top = self.omega * dis + p_a
            b = torch.topk(p, k=2, dim=1)[1][:, 1]

            for _ in range(tau):
                p = F.softmax(model(input_clone), dim=1)
                if p[0][a] >= top:
                    break
                clamped_p = torch.clamp(p[0][a], min=p_a, max=top)
                loss = -0.1 * torch.log(clamped_p)
                optimizer.zero_grad()
                grads = torch.autograd.grad(loss, input_clone, retain_graph=True)
                input_clone.data = input_clone.data - lr * grads[0]
        return input_clone.detach()

    @timer
    def evolved_fingerprint_algorithm(self, model: Module, component: str):
        primary_data_path = f"./fingerprint/{self.domain}/primary_{component}.pkl"
        data_set = load_result(primary_data_path)
        sample_record = []
        label_record = []
        # init progress bar
        tqdm_bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        tqdm_kwargs = {
            "desc": "EFA",
            "bar_format": tqdm_bar_format,
            "ncols": 80,
            "bar_format": "\033[32m" + tqdm_bar_format + "\033[0m",
            "leave": False,
        }
        bar = tqdm(range(self.n), **tqdm_kwargs)
        for i in bar:
            s = self.efa(
                model=model,
                input=data_set["data"][i],
                component=component,
                tau=self.tau,
                lr=self.lr,
            )
            sample_record.append(s)
            label_record.append(data_set["label"][i])
        new_inputs = torch.cat(sample_record, dim=0)
        new_labels = torch.argmax(model(new_inputs), dim=-1)
        save_result(
            f"./fingerprint/{self.domain}/evolved_{component}_original.pkl",
            {"data": new_inputs, "label": new_labels},
        )


if __name__ == "__main__":
    ef = EvolvedFingerprint(
        n=80,
        lr=0.001,
        tau=10,
        mu=1e-5,
        omega=2,
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
        domain="CIFAR10",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    source = ModelLoader.domain_ada("CIFAR10").load_model(index=0, model_type="source")
    ef.evolved_fingerprint_algorithm(model=source, component="cc")
    ef.evolved_fingerprint_algorithm(model=source, component="wc")
    ef.evolved_fingerprint_algorithm(model=source, component="cu")
    ef.evolved_fingerprint_algorithm(model=source, component="wu")
