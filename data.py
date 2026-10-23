"""Character-level TinyShakespeare: small enough to train on a laptop CPU."""

import os
import urllib.request

import torch

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class CharDataset:
    def __init__(self, val_frac: float = 0.1):
        path = os.path.join(DATA_DIR, "tinyshakespeare.txt")
        if not os.path.exists(path):
            os.makedirs(DATA_DIR, exist_ok=True)
            print(f"downloading {URL}")
            urllib.request.urlretrieve(URL, path)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(len(data) * (1 - val_frac))
        self.train, self.val = data[:n], data[n:]

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def batch(self, split: str, batch_size: int, block_size: int, generator=None):
        data = self.train if split == "train" else self.val
        ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
        return x, y
