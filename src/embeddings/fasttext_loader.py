"""Build the embedding matrix: row i = fastText vector for vocabulary word i."""

import gzip
import json
from collections import Counter

import numpy as np

from src.config import EMB_DIM, FASTTEXT_PATH, PROCESSED_DIR, RANDOM_SEED
from src.data.preprocessing import clean_arabic_text
from src.data.vocabulary import Vocabulary


def build_embedding_matrix(vocab: Vocabulary, vec_path=FASTTEXT_PATH,
                           token_counts: Counter | None = None, dim: int = EMB_DIM,
                           seed: int = RANDOM_SEED) -> tuple[np.ndarray, dict]:
    """Stream a fastText .vec(.gz) file and return ((vocab_size, dim) float32 matrix, stats).

    fastText keys are normalised with the same clean_arabic_text as our data; the file is
    frequency-sorted, so the first match for a vocab word wins. Unmatched words get small
    random vectors (std estimated from the matched vectors); the <pad> row stays zeros.
    """
    matrix = np.zeros((len(vocab), dim), dtype=np.float32)
    found = np.zeros(len(vocab), dtype=bool)
    value_sum, value_sumsq, value_count = 0.0, 0.0, 0

    open_fn = gzip.open if str(vec_path).endswith(".gz") else open
    with open_fn(vec_path, "rt", encoding="utf-8", errors="ignore") as f:
        f.readline()  # header: "<n_words> <dim>"
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) != dim + 1:
                continue
            idx = vocab.stoi.get(parts[0])
            if idx is None:
                cleaned = clean_arabic_text(parts[0]).split()
                if len(cleaned) != 1:
                    continue
                idx = vocab.stoi.get(cleaned[0])
            if idx is None or found[idx]:
                continue
            vec = np.asarray(parts[1:], dtype=np.float32)
            matrix[idx] = vec
            found[idx] = True
            value_sum += vec.sum(dtype=np.float64)
            value_sumsq += np.square(vec, dtype=np.float64).sum()
            value_count += dim

    std = float(np.sqrt(value_sumsq / value_count - (value_sum / value_count) ** 2)) \
        if value_count else 0.05
    missing = ~found
    missing[vocab.pad_id] = False  # <pad> stays a zero vector
    rng = np.random.default_rng(seed)
    matrix[missing] = rng.normal(0.0, std, size=(int(missing.sum()), dim)).astype(np.float32)

    stats = {
        "vocab_size": len(vocab),
        "found": int(found.sum()),
        "missing": int(missing.sum()),
        "type_coverage": float(found.sum() / len(vocab)),
        "random_init_std": std,
    }
    if token_counts:
        covered = sum(c for w, c in token_counts.items()
                      if (i := vocab.stoi.get(w)) is not None and found[i])
        total = sum(token_counts.values())
        stats["token_coverage"] = covered / total if total else 0.0
    return matrix, stats


if __name__ == "__main__":
    import pandas as pd

    vocab = Vocabulary.load(PROCESSED_DIR / "vocab.json")
    print(f"vocabulary loaded: {len(vocab)} words")

    train_counts = Counter()
    for tokens in pd.read_csv(PROCESSED_DIR / "train.csv")["clean"].fillna("").str.split():
        train_counts.update(tokens)

    matrix, stats = build_embedding_matrix(vocab, token_counts=train_counts)
    out_path = PROCESSED_DIR / "embedding_matrix.npy"
    np.save(out_path, matrix)
    (PROCESSED_DIR / "embedding_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nembedding matrix: {matrix.shape} {matrix.dtype} -> {out_path}")
    print(f"words found in fastText: {stats['found']}/{stats['vocab_size']} "
          f"({stats['type_coverage']:.1%} of vocabulary types)")
    print(f"train tokens covered: {stats['token_coverage']:.2%} (weighted by frequency)")
    print(f"missing words -> random vectors with std {stats['random_init_std']:.4f}")
    print(f"pad row all zeros: {not matrix[vocab.pad_id].any()}")
    print(f"spot check - vector for 'الله' is non-zero: {bool(matrix[vocab.stoi['الله']].any())}")
