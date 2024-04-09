import os
import argparse

import torch
from torch.nn import Module
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torcheeg.model_selection import KFoldGroupbyTrial
from torch.utils.tensorboard import SummaryWriter


import utils
import data_loader
from model_loader import ModelLoader


def train(
    model: Module,
    data_loader: DataLoader,
    loss_fn: CrossEntropyLoss,
    optimizer: Adam,
) -> float:
    total_batches = len(data_loader)
    loss_record = []
    #
    model.train()
    for _, batch_data in enumerate(data_loader):
        b_x = batch_data[0].to(device)
        b_y = batch_data[1].to(device)
        output = model(b_x)
        loss = loss_fn(output, b_y)
        #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #
        loss = loss.detach().item()
        loss_record.append(loss)
    mean_train_loss = sum(loss_record) / total_batches
    return mean_train_loss


def test(model: Module, data_loader: DataLoader, loss_fn: CrossEntropyLoss):
    total_sample_num = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    test_loss, correct_sample_num = 0, 0
    with torch.no_grad():
        for batch_data in data_loader:
            b_x = batch_data[0].to(device)
            b_y = batch_data[1].to(device)
            output = model(b_x)
            test_loss += loss_fn(output, b_y).item()

            correct_sample_num += (
                (output.argmax(1) == b_y).type(torch.float).sum().item()
            )
    #
    test_loss /= num_batches
    #
    test_accuracy = correct_sample_num / total_sample_num
    return test_loss, test_accuracy


def fit(
    index: int,
    model: Module,
    kfold: KFoldGroupbyTrial,
    dataset: Dataset,
    args: argparse.ArgumentParser,
):
    model.to(device)
    #
    if index < 5:
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = Adam(
            model.fc.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    loss_fn = CrossEntropyLoss()
    #
    save_dir = f"./model/finetune/{args.model_name}/finetune_{index}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(save_dir.replace("/model/", "/result/"))
    best_acc = 0
    for split_idx, (train_dataset, test_dataset) in enumerate(kfold.split(dataset)):
        #
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
        )
        #
        best_test_acc = 0
        for epoch_id in range(args.epochs):
            #
            train_loss = train(
                model=model,
                data_loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
            )
            writer.add_scalar(
                "Loss/Train", train_loss, (split_idx * args.epochs) + epoch_id
            )
            #
            test_loss, test_acc = test(
                model=model, data_loader=test_loader, loss_fn=loss_fn
            )
            writer.add_scalar(
                "Loss/Test", test_loss, (split_idx * args.epochs) + epoch_id
            )
            #
            if test_acc > best_test_acc:
                best_test_acc = test_acc

                torch.save(
                    model.state_dict(), os.path.join(save_dir, f"model_{split_idx}.pth")
                )
        #
        model.load_state_dict(
            torch.load(os.path.join(save_dir, f"model_{split_idx}.pth"))
        )
        _, test_acc = test(model, test_loader, loss_fn)
        writer.add_scalar("Acc/Test", test_acc, split_idx + 1)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "model_best.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="eegconformer")
    parser.add_argument("--gpu", type=int)
    parser.add_argument(
        "--label", type=str, default="valence", choices=["valence", "arousal"]
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    args = parser.parse_args()

    #
    utils.seed_everything(2023)
    #
    if torch.cuda.is_available():
        device = torch.device("cuda", args.gpu)

    _, attack_dataset = data_loader.load_split_dataset(
        model_name=args.model_name, label=args.label
    )
    #
    kfold = KFoldGroupbyTrial(
        n_splits=5, shuffle=True, split_path=f"./data/DEAP/{args.model_name}/attack"
    )
    #
    for index in range(0, 10):
        #
        model = ModelLoader.domain_ada("DEAP").load_model(mode="source")
        fit(
            index=index,
            model=model,
            kfold=kfold,
            dataset=attack_dataset,
            args=args,
        )
