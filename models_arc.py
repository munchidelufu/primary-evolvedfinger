import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForSequenceClassification
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

"""
NLP Models
"""


class Bert(nn.Module):
    def __init__(
        self,
        classes=10,
        hidden_size=768,
    ):
        super(Bert, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            "bert-base-chinese", num_labels=hidden_size
        )
        for param in self.bert.parameters():
            param.requires_grad = True
        self.lin = nn.Linear(hidden_size, classes)

    def forward(self, x):
        sequence_classifier_output = self.bert(**x)
        out = self.lin(sequence_classifier_output.logits)
        return out


class DPCNN(nn.Module):
    """Deep Pyramid Convolutional Neural Networks for Text Categorization"""

    def __init__(self, vocab_size=21128, embed=768, num_filters=250, num_classes=10):
        super(DPCNN, self).__init__()
        self.vocab_size = vocab_size
        self.embed = embed
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.embedding = nn.Embedding(self.vocab_size, self.embed)
        self.conv_region = nn.Conv2d(1, self.num_filters, (3, self.embed), stride=1)
        self.conv = nn.Conv2d(self.num_filters, self.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.lin = nn.Linear(self.num_filters, self.num_classes)

    def forward(self, x):
        x = x["input_ids"]
        x = self.embedding(x)
        x = x.unsqueeze(1)  # [batch_size, 250, seq_len, 1]
        x = self.conv_region(x)  # [batch_size, 250, seq_len-3+1, 1]

        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze()  # [batch_size, num_filters(250)]
        x = self.lin(x)
        return x

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        x = x + px
        return x


class TextCNN(nn.Module):
    """Convolutional Neural Networks for Sentence Classification"""

    def __init__(
        self,
        vocab_size=21128,
        embed=768,
        num_filters=256,
        filter_sizes=(2, 3, 4),
        drop_out=0.5,
        num_classes=10,
    ):
        super(TextCNN, self).__init__()
        self.vocab_size = vocab_size
        self.embed = embed
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.num_classes = num_classes
        self.embedding = nn.Embedding(self.vocab_size, self.embed)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, self.embed)) for k in self.filter_sizes]
        )
        self.dropout = nn.Dropout(drop_out)
        self.lin = nn.Linear(
            self.num_filters * len(self.filter_sizes), self.num_classes
        )

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x["input_ids"])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.lin(out)
        return out


class TextRCNN(nn.Module):
    """Recurrent Convolutional Neural Networks for Text Classification"""

    def __init__(
        self,
        vocab_size=21128,
        embed=768,
        hidden_size=256,
        num_layers=1,
        dropout=0.5,
        pad_size=32,
        num_classes=10,
    ):
        super(TextRCNN, self).__init__()
        self.vocab_size = vocab_size
        self.embed = embed
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.pad_size = pad_size
        self.num_classes = num_classes
        self.embedding = nn.Embedding(self.vocab_size, self.embed)
        self.lstm = nn.LSTM(
            self.embed,
            self.hidden_size,
            self.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=self.dropout,
        )
        self.maxpool = nn.MaxPool1d(self.pad_size)
        self.fc = nn.Linear(self.hidden_size * 2 + self.embed, self.num_classes)

    def forward(self, x):
        embed = self.embedding(
            x["input_ids"]
        )  # [batch_size, seq_len, embeding]=[64, 32, 64]
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out


class TextRNN(nn.Module):
    """Recurrent Neural Network for Text Classification with Multi-Task Learning"""

    def __init__(
        self,
        vocab_size=21128,
        embed=768,
        hidden_size=128,
        num_layers=2,
        dropout=0.5,
        num_classes=10,
    ):
        super(TextRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embed = embed
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.embedding = nn.Embedding(self.vocab_size, self.embed)
        self.lstm = nn.LSTM(
            self.embed,
            self.hidden_size,
            self.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(self.hidden_size * 2, self.num_classes)

    def forward(self, x):
        out = self.embedding(
            x["input_ids"]
        )  # [batch_size, seq_len, embeding]=[128, 32, 768]
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  #
        return out


"""
BCI Models
"""


## Conformer
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (22, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d(
                (1, 75), (1, 15)
            ),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(
                40, emb_size, (1, 1), stride=(1, 1)
            ),  # transpose, conv could enhance fiting ability slightly
            Rearrange("b e (h) (w) -> b (h w) e"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhal, bhlv -> bhav ", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        emb_size,
        num_heads=10,
        drop_p=0.5,
        forward_expansion=4,
        forward_drop_p=0.5,
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, num_heads, drop_p),
                    nn.Dropout(drop_p),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(
                        emb_size, expansion=forward_expansion, drop_p=forward_drop_p
                    ),
                    nn.Dropout(drop_p),
                )
            ),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()

        # global average pooling
        self.clshead = nn.Sequential(
            Reduce("b n e -> b e", reduction="mean"),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes),
        )
        self.fc = nn.Sequential(
            nn.Linear(880, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out


class Conformer(nn.Module):
    def __init__(self, emb_size=40, depth=6, n_classes=4, **kwargs):
        super().__init__()
        self.patchEmbedding = PatchEmbedding(emb_size)
        self.transformerEncoder = TransformerEncoder(depth, emb_size)
        self.fc = ClassificationHead(emb_size, n_classes)

    def forward(self, x):
        x = self.patchEmbedding(x)
        x = self.transformerEncoder(x)
        x = self.fc(x)
        return x


# DeepNet
class DeepConvNet(nn.Module):
    def __init__(self, num_classes, Chans=32, dropoutRate=0.5):
        super(DeepConvNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 2)),
            nn.Conv2d(25, 25, kernel_size=(Chans, 1)),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(dropoutRate),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 2)),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(dropoutRate),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 2)),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(dropoutRate),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5)),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(dropoutRate),
        )
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


# ShallowNet
class Square(nn.Module):
    def forward(self, x):
        return torch.square(x)


class Log(nn.Module):
    def forward(self, x):
        return torch.log(torch.clamp(x, min=1e-7, max=10000))


class ShallowConvNet(nn.Module):
    def __init__(self, num_classes, Chans=32, dropoutRate=0.5):
        super(ShallowConvNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=(1, 13)),
            nn.Conv2d(40, 40, kernel_size=(Chans, 1), bias=False),
            nn.BatchNorm2d(40, eps=1e-05, momentum=0.9),
            Square(),
            nn.AvgPool2d(kernel_size=(1, 7), stride=(1, 3)),
            Log(),
            nn.Dropout(dropoutRate),
        )
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(1480, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x
