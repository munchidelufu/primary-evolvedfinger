import os
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torcheeg.model_selection import KFoldGroupbyTrial

import utils
import data_loader
from acmmm24_811.models_arc import DeepConvNet
from acmmm24_811.models_arc import ShadowNet
from model_loader import ModelLoader


def train(
    tea_model: Module,
    stu_model: Module,
    data_loader: DataLoader,
    loss_fn: Module,
    optimizer: Adam,
) -> float:
    total_batches = len(data_loader)
    tea_model.to(device)
    stu_model.to(device)
    #
    tea_model.eval()
    stu_model.train()

    loss_record = []
    for batch_id, batch_data in enumerate(data_loader):
        b_x = batch_data[0].to(device)
        b_y = batch_data[1].to(device)

        t_output = tea_model(b_x)
        pred = t_output.argmax(1)
        s_output = stu_model(b_x)
        loss = loss_fn(s_output, pred)
        #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #
        loss = loss.detach().item()
        loss_record.append(loss)
    mean_train_loss = sum(loss_record) / total_batches
    return mean_train_loss


def val(model: Module, data_loader: DataLoader, loss_fn: Module):
    total_sample_num = len(data_loader.dataset)
    model.eval()
    correct_sample_num = 0
    epoch_loss = 0
    batch_num = len(data_loader)
    with torch.no_grad():
        for batch_data in data_loader:
            b_x = batch_data[0].to(device)
            b_y = batch_data[1].to(device)
            output = model(b_x)
            loss = loss_fn(output, b_y)
            epoch_loss += loss.detach().item()
            correct_sample_num += (
                (output.argmax(1) == b_y).type(torch.float).sum().item()
            )
    #
    test_accuracy = correct_sample_num / total_sample_num
    loss = round(epoch_loss / batch_num, 7)
    return loss, test_accuracy


def fit(
    index: int,
    tea_model: Module,
    stu_model: Module,
    dataset: Dataset,
    kfold: KFoldGroupbyTrial,
    args: argparse.ArgumentParser,
):
    tea_model.to(device)
    stu_model.to(device)
    #
    optimizer = Adam(stu_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = ReduceLROnPlateau(
        optimizer=optimizer, mode="min", factor=0.1, patience=5, verbose=True
    )
    loss_fn = CrossEntropyLoss()
    #
    save_dir = (
        f"./model/surrogate/{args.teacher_name}_{args.student_name}/surrogate_{index}"
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(save_dir.replace("/model/", "/result/"))
    best_acc = 0
    for split_idx, (train_dataset, test_dataset) in enumerate(kfold.split(dataset)):
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
        )
        #
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
                tea_model=tea_model,
                stu_model=stu_model,
                data_loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epoch_id=epoch_id,
            )
            writer.add_scalar(
                "Loss/Train", train_loss, (split_idx * args.epochs) + epoch_id
            )
            #
            test_loss, test_acc = val(
                model=stu_model, data_loader=test_loader, loss_fn=loss_fn
            )
            writer.add_scalar(
                "Loss/Test", test_loss, (split_idx * args.epochs) + epoch_id
            )
            # 更新学习率
            lr_scheduler.step(test_loss)
            #
            if test_acc > best_test_acc:
                best_test_acc = test_acc

                torch.save(
                    stu_model.state_dict(),
                    os.path.join(save_dir, f"model_{split_idx}.pth"),
                )
        #
        stu_model.load_state_dict(
            torch.load(os.path.join(save_dir, f"model_{split_idx}.pth"))
        )
        loss, test_acc = val(stu_model, test_loader, loss_fn=loss_fn)
        writer.add_scalar("Acc/Test", test_acc, split_idx + 1)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(stu_model.state_dict(), os.path.join(save_dir, "model_best.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_name", type=str, default="eegconformer")
    parser.add_argument("--student_name", type=str, choices=["deepnet", "shallownet"])
    parser.add_argument("--gpu", type=int)
    parser.add_argument(
        "--label", type=str, default="valence", choices=["valence", "arousal"]
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.9)
    args = parser.parse_args()

    #
    utils.seed_everything(2023)
    #
    if torch.cuda.is_available():
        device = torch.device("cuda", args.gpu)
    #
    source_model = ModelLoader.domain_ada("DEAP").load_model(mode="source")
    #
    #
    train_dataset, attack_dataset = data_loader.load_split_dataset(
        model_name=args.model_name, label=args.label
    )
    #
    # pair_kfold
    kfold = KFoldGroupbyTrial(
        n_splits=5, split_path=f"./data/{args.teacher_name}/train"
    )

    for index in range(2):
        model = DeepConvNet(num_classes=2)
        fit(
            index=index,
            tea_model=source_model,
            stu_model=model,
            dataset=train_dataset,
            kfold=kfold,
            args=args,
        )
