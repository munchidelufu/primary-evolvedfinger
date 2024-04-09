from collections import defaultdict

import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset

import utils
import thulac


class DetectionAE(nn.Module):
    def __init__(self) -> None:
        super(DetectionAE, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # Adding pooling layer
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # Adding pooling layer
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # Adding pooling layer
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, 3, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class QueryAttack:
    def __init__(self) -> None:
        if torch.cuda.is_available():
            self.device = torch.device("cuda", 4)

    def key_sample_detection(self, sample, threshold: float = 55, verbose: bool = True):
        model = DetectionAE()
        model.load_state_dict(torch.load(f"./model/autoencoder/model_best.pth"))
        model.eval()
        model.to(self.device)
        sample = torch.from_numpy(sample) if isinstance(sample, np.ndarray) else sample
        sample = torch.unsqueeze(sample, dim=0) if len(sample.shape) != 4 else sample
        sample = sample.to(self.device)
        total_loss = 0.0
        output = model(sample).detach()
        total_loss += torch.norm((output - sample), p=2).item()
        if verbose:
            return total_loss
        if total_loss >= threshold:
            return True
        else:
            return False

    def quary_attack(self, dataset):
        model = DetectionAE()
        model.load_state_dict(torch.load(f"./model/autoencoder/model_best.pth"))
        model.eval()
        model.to(self.device)
        inputs = []
        labels = []
        for sample in dataset:
            b_x = torch.unsqueeze(sample[0], dim=0).to(self.device)
            ae_output = model(b_x).detach()
            inputs.append(torch.squeeze(ae_output))
            labels.append(sample[1])
        inputs = torch.stack(inputs)
        labels = torch.stack(labels)
        return TensorDataset(inputs, labels)


class InputSmooth:

    @staticmethod
    def denoise_eeg(path: str, save_path: str):
        data = []
        data_set = utils.load_result(path)
        eegs, min_val, max_val = utils.denormalize(data_set["data"])
        eegs = np.array(eegs.detach().cpu()) if isinstance(eegs, torch.Tensor) else eegs
        for eeg in eegs:
            fractional, integer = np.modf(eeg * 255)
            eeg = (integer).astype(np.uint8)
            blurred_image_mean = cv2.blur(eeg, (5, 20))
            data.append((blurred_image_mean + fractional) / 200)
        data = torch.from_numpy(np.array(data)).float()
        data = utils.normalize(data, min_val, max_val)
        utils.save_result(save_path, {"data": data, "label": data_set["label"]})


class ADJAttack:

    def __init__(self, path: str) -> None:
        self.thu1 = thulac.thulac()
        self.path = path
        self.total = self.count_adj()
        self.total_synonyms = {
            "男": "男性",
            "大": "宽敞",
            "新": "崭新的",
            "高": "高大",
            "女": "女性",
            "金": "金色",
            "广": "广袤的",
            "小": "渺小",
            "香": "鲜香",
            "通": "通常",
            "老": "老旧",
            "黄": "黄色",
            "银": "银色",
            "顺": "顺利的",
            "热": "滚热的",
            "多": "丰富",
            "长": "不短",
            "黑": "黑色",
            "青": "青色",
            "富": "有钱",
            "实": "实在",
            "白": "白色",
            "密": "密集的",
        }

    def find_adjectives(self, sentence):
        text = self.thu1.cut(sentence, text=True)
        adjectives = []
        for token in text.split(" "):
            try:
                word, tag = token.split("_")
                if tag.startswith("a"):
                    adjectives.append(word)
            except:
                pass
        return adjectives

    def get_synonyms(self, word):
        return self.total_synonyms.get(word, "")

    def replace_adjectives_with_synonyms(self, query_sentence):
        res_sentence = query_sentence
        adjectives = self.find_adjectives(res_sentence)
        for adj in adjectives:
            synonyms = self.get_synonyms(adj)
            if synonyms:
                res_sentence = res_sentence.replace(adj, synonyms)

        return res_sentence

    def count_adj(self):
        l = []
        total = defaultdict()
        with open(self.path, "r") as file:
            for line in file.readlines():
                l.extend(self.find_adjectives(line[0]))
        for w in l:
            total[w] = total.get(w, 0) + 1
        return dict(sorted(total.items(), key=lambda x: x[1], reverse=True))
