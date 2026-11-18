"""Torch Dataset and DataLoader factory for the frozen splits."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import LABEL2ID, MAX_LEN, PROCESSED_DIR, RANDOM_SEED
from src.data.vocabulary import Vocabulary


class ArabicTextDataset(Dataset):
    """One item = (ids[max_len], length, label) as int64 tensors; ids encoded once in __init__."""

    def __init__(self, df: pd.DataFrame, vocab: Vocabulary, max_len: int = MAX_LEN):
        self.max_len = max_len
        self.pad_id = vocab.pad_id
        self.ids = [np.array(vocab.encode(text.split())[:max_len], dtype=np.int64)
                    for text in df["clean"].fillna("")]
        self.labels = [LABEL2ID[label] for label in df["label"]]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids = self.ids[idx]
        length = max(len(ids), 1)
        padded = np.full(self.max_len, self.pad_id, dtype=np.int64)
        padded[: len(ids)] = ids
        return (
            torch.from_numpy(padded),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def make_loaders(batch_size: int = 64, seed: int = RANDOM_SEED,
                 max_len: int = MAX_LEN) -> dict[str, DataLoader]:
    """One DataLoader per split; shuffle train only, with a seeded generator."""
    vocab = Vocabulary.load(PROCESSED_DIR / "vocab.json")
    loaders = {}
    for name in ["train", "val", "test"]:
        df = pd.read_csv(PROCESSED_DIR / f"{name}.csv")
        ds = ArabicTextDataset(df, vocab, max_len)
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            generator=torch.Generater().manual_seed(seed) if name == "train" else None,
        )
    return loaders
