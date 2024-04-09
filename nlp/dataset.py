import os
import pickle as pkl
from tqdm import tqdm

from torch.utils.data import Dataset

from detect_erase_attack import ADJAttack


class TextDataSet(Dataset):
    def __init__(self, path, tokenizer, pad_size=32, attack: str = None) -> None:
        super().__init__()

        self.path = path
        cache_file = "".join(path.split("/")[:-1]) + f"/{attack}_cache"
        if attack and os.path.exists(cache_file):
            self.dataset = pkl.load(cache_file)
        else:
            self.tokenizer = tokenizer
            self.attack = None
            if attack == "adj":
                self.attack = ADJAttack(path=path)
            if "THU" in self.path:
                self.pad_size = 32
            elif "Onlineshop" in self.path:
                self.pad_size = 64
                self.cal = [
                    "书籍",
                    "洗发水",
                    "热水器",
                    "平板",
                    "蒙牛",
                    "衣服",
                    "手机",
                    "计算机",
                    "水果",
                    "酒店",
                ]
            self.dataset = self._load_data()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def _load_data(self):
        contents = []

        count = 0
        with open(self.path, mode="r", encoding="UTF-8") as f:
            for line in tqdm(f):
                line = line.strip()
                if not line:
                    continue
                if "THU" in self.path:
                    content, label = line.split("\t")
                elif "Onlineshop" in self.path:
                    line = line.split(",")
                    if len(line) == 3 and line[0] != "cat":
                        cat, _, content = line
                        label = self.cal.index(cat)
                    else:
                        continue
                if self.attack:
                    n_content = self.attack.replace_adjectives_with_synonyms(content)
                    print(n_content == content)
                    content = n_content
                encoded_input = self.tokenizer(
                    content,
                    padding="max_length",
                    truncation=True,
                    max_length=self.pad_size,
                    return_tensors="pt",
                )
                try:
                    contents.append((encoded_input, int(label)))
                except ValueError:
                    print(label, type(label))
        return contents
