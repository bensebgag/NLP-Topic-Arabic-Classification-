import json
from collections import Counter
from pathlib import Path

from src.config import PAD_TOKEN, UNK_TOKEN, MIN_FREQ


class Vocabulary:
    pad_id = 0
    unk_id = 1

    def __init__(self, itos):
        self.IndextoString = list(itos)
        self.StringtoIndex = {w: i for i, w in enumerate(self.IndextoString)}

    @classmethod
    def build(cls, token_lists, min_freq=MIN_FREQ):
        counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)
        words = [w for w, c in sorted(counter.items(), key=lambda x: (-x[1], x[0]))
                 if c >= min_freq]
        return cls([PAD_TOKEN, UNK_TOKEN] + words)

    def encode(self, tokens):
        return [self.StringtoIndex.get(t, self.unk_id) for t in tokens]

    def __len__(self):
        return len(self.IndextoString)

    def __contains__(self, word):
        return word in self.StringtoIndex

    def save(self, path):
        Path(path).write_text(json.dumps(self.IndextoString, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


if __name__ == "__main__":
    import pandas as pd
    from src.config import PROCESSED_DIR

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    token_lists = train_df["clean"].fillna("").str.split()

    vocab = Vocabulary.build(token_lists)
    vocab.save(PROCESSED_DIR / "vocab.json")
    print(f"vocabulary size: {len(vocab)} (incl. {PAD_TOKEN} and {UNK_TOKEN}, min_freq={MIN_FREQ})")

    print("\ntop 10 most frequent words:")
    for word in vocab.IndextoString[2:12]:
        print(f"  id {vocab.StringtoIndex[word]:>6}  {word}")

    example = "ارتفع المؤشر العام لسوق ابوظبي للاوراق الماليه امس بنسبه 0.14 ووصل الي 2859 نقطه".split()
    print("\nexample encoding (the Finance sentence from the teaching script):")
    print(example)
    print(vocab.encode(example))

    val_df = pd.read_csv(PROCESSED_DIR / "val.csv")
    total = oov = 0
    for tokens in val_df["clean"].fillna("").str.split():
        total += len(tokens)
        oov += sum(1 for t in tokens if t not in vocab)
    print(f"\nOOV rate on validation: {oov}/{total} tokens = {oov/total:.2%}")
    print("(test OOV will be measured once, at the final evaluation)")
