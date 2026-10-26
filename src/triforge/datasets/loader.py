"""Memory-mapped token dataset with resumable random sampling."""

from __future__ import annotations

import os

import numpy as np
import torch


class TokenDataset:
    """Samples (x, y, mask) windows from a packed token stream.

    The arrays are memory-mapped, so a multi-GB shard costs almost no RAM. The sampler is a
    seeded torch.Generator whose state is saved in checkpoints, so a resumed run sees exactly
    the batches it would have seen without the interruption.
    """

    def __init__(self, directory: str, split: str, seed: int = 0):
        self.tokens = np.load(os.path.join(directory, f"{split}.tokens.npy"), mmap_mode="r")
        mpath = os.path.join(directory, f"{split}.mask.npy")
        self.mask = np.load(mpath, mmap_mode="r") if os.path.exists(mpath) else None
        self.gen = torch.Generator().manual_seed(seed)

    def __len__(self):
        return int(self.tokens.shape[0])

    def batch(self, batch_size: int, seq_len: int, device="cpu", generator=None):
        g = generator or self.gen
        hi = len(self) - seq_len - 1
        if hi <= 0:
            raise ValueError(f"split has {len(self)} tokens, fewer than seq_len={seq_len}")
        ix = torch.randint(hi, (batch_size,), generator=g).tolist()
        x = np.stack([self.tokens[i:i + seq_len] for i in ix]).astype(np.int64)
        y = np.stack([self.tokens[i + 1:i + 1 + seq_len] for i in ix]).astype(np.int64)
        xt, yt = torch.from_numpy(x), torch.from_numpy(y)
        m = None
        if self.mask is not None:
            m = torch.from_numpy(np.stack([self.mask[i + 1:i + 1 + seq_len] for i in ix]).astype(np.float32))
        if device != "cpu":
            xt, yt = xt.to(device, non_blocking=True), yt.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True) if m is not None else None
        return xt, yt, m

    def state(self):
        return self.gen.get_state()

    def set_state(self, st):
        self.gen.set_state(st)
