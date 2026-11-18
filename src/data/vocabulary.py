"""Vocabulary: word <-> id mapping, built from the train split only."""

import json
from collections import Counter
from pathlib import Path

from src.config import MIN_FREQ, PAD_TOKEN, PROCESSED_DIR, UNK_TOKEN


class Vocabulary:
    """Word <-> id mapping built once from the train split, then saved and reloaded."""

    pad_id = 0
    unk_id = 1

    def __init__(self, itos: list[str]):
        self.itos = list(itos)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def build(cls, token_lists, min_freq: int = MIN_FREQ) -> "Vocabulary":
        """Count tokens, keep words with freq >= min_freq, ordered by (-count, word)."""
        counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)
        words = [w for w, c in sorted(counter.items(), key=lambda x: (-x[1], x[0]))
                 if c >= min_freq]
        return cls([PAD_TOKEN, UNK_TOKEN] + words)

    def encode(self, tokens: list[str]) -> list[int]:
        """Map tokens to ids; unseen words become <unk> (id 1)."""
        return [self.stoi.get(t, self.unk_id) for t in tokens]

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, word: str) -> bool:
        return word in self.stoi

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.itos, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Vocabulary":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


if __name__ == "__main__":
    import pandas as pd

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    vocab = Vocabulary.build(train_df["clean"].fillna("").str.split())
    vocab.save(PROCESSED_DIR / "vocab.json")
    print(f"vocabulary size: {len(vocab)} (min_freq={MIN_FREQ}); "
          f"saved to {PROCESSED_DIR / 'vocab.json'}")
