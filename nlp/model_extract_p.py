import os
import argparse

import torch
import importlib
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
from torch.utils.data import DataLoader

import utils
from models_arc import Bert
from models_arc import TextRNN
from dataset import TextDataSet
from model_loader import ModelLoader


def collate_fn(batch):
    encode_inputs, labels = zip(*batch)
    input_ids = torch.stack([s["input_ids"][0] for s in list(encode_inputs)], dim=0).to(
        device
    )
    token_type_ids = torch.stack(
        [s["token_type_ids"][0] for s in list(encode_inputs)], dim=0
    ).to(device)
    attention_mask = torch.stack(
        [s["attention_mask"][0] for s in list(encode_inputs)], dim=0
    ).to(device)
    encode_inputs = {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
    }
    labels = torch.tensor(list(labels)).to(device)
    return encode_inputs, labels


def train(
    index,
    tea_model,
    stu_model,
    train_loader,
    dev_loader,
    optimizer,
):
    tea_model.to(device)
    stu_model.train()
    stu_model.to(device)
    total_batch = 0
    dev_best_loss = float("inf")
    last_improve = 0
    flag = False
    save_dir = f"./model/THUCNews/{args.model_type}/"
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(args.epochs):
        for i, (datas, labels) in enumerate(train_loader):
            t_outputs = tea_model(datas)
            pred = t_outputs.argmax(1)
            s_outputs = stu_model(datas)
            loss = nn.KLDivLoss()(
                F.softmax(s_outputs / args.T, dim=1),
                F.softmax(t_outputs / args.T, dim=1),
            ) * (args.alpha) + (1 - args.alpha) * F.cross_entropy(s_outputs, pred)
            # Clear and update gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                train_acc = (s_outputs.argmax(1) == labels).sum().item() / len(labels)
                dev_acc, dev_loss = test(stu_model, dev_loader)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(
                        stu_model.state_dict(),
                        os.path.join(save_dir, f"{args.stu_name}_{index}.ckpt"),
                    )
                    last_improve = total_batch
                stu_model.train()
            total_batch += 1
            if total_batch - last_improve > 1000:
                flag = True
                break
        if flag:
            break


def test(model, data_loader):
    model.eval()
    correct_num, total_num, total_loss = 0, 0, 0
    with torch.no_grad():
        for datas, labels in data_loader:
            outputs = model(datas)
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item()
            correct_num += (outputs.argmax(1) == labels).sum().item()
            total_num += len(labels)
    acc = correct_num / total_num
    loss = total_loss / len(data_loader)
    return acc, loss


def fit(index, tea_model, stu_model, train_loader, dev_loader, test_loader):
    tea_model.eval()

    optimizer = torch.optim.Adam(stu_model.parameters(), lr=args.learning_rate)

    train(
        index,
        tea_model,
        stu_model,
        train_loader,
        dev_loader,
        optimizer,
    )
    stu_model.load_state_dict(
        torch.load(
            f"./model/THUCNews/{args.model_type}/{args.stu_name}_{index}.ckpt",
            map_location=device,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tea_name", type=str, default="bert", help="choose a model")
    parser.add_argument("--stu_name", type=str, choices=["TextCNN", "TextRNN", "DPCNN"])
    parser.add_argument("--model_type", type=str, default="model_extract_p_trigger")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.9)
    args = parser.parse_args()

    utils.seed_everything(2023)
    if torch.cuda.is_available():
        device = torch.device("cuda", args.gpu)
    # tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    #  preprocess data
    train_path = f"./data/THUCNews/source.txt"
    dev_path = f"./data/THUCNews/dev.txt"
    test_path = f"./data/THUCNews/test.txt"
    train_data = TextDataSet(train_path, tokenizer)
    dev_data = TextDataSet(dev_path, tokenizer)
    test_data = TextDataSet(test_path, tokenizer)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    tea_model = ModelLoader.domain_ada("THUCNews").load_model(mode="source")

    #
    for index in range(5):
        model = importlib.import_module(f"models.{args.stu_name}")
        stu_model = TextRNN()
        fit(index, tea_model, stu_model, train_loader, dev_loader, test_loader)
