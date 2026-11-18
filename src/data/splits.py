"""De-duplication and the stratified train/val/test split (torch-free)."""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import HOLDOUT_FRAC, PROCESSED_DIR, RANDOM_SEED


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop repeated `clean` texts (keep first); returns (deduplicated df, stats)."""
    n_labels_per_clean = df.groupby("clean")["label"].nunique()
    assert (n_labels_per_clean <= 1).all(), "duplicate articles with conflicting labels"
    before = len(df)
    df = df.drop_duplicates(subset="clean", keep="first").reset_index(drop=True)
    return df, {"before": before, "after": len(df), "removed": before - len(df)}


def split_dataset(df: pd.DataFrame, seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/15/15 split on `label`; returns (train_df, val_df, test_df)."""
    train_df, rest_df = train_test_split(
        df, test_size=HOLDOUT_FRAC, stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        rest_df, test_size=0.5, stratify=rest_df["label"], random_state=seed
    )
    return train_df, val_df, test_df


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                out_dir=PROCESSED_DIR) -> None:
    """Freeze the splits as CSVs; later steps read these instead of recomputing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)


if __name__ == "__main__":
    from src.data.loader import load_dataset
    from src.data.preprocessing import clean_arabic_text

    df = load_dataset()
    df["clean"] = df["text"].apply(clean_arabic_text)
    df, stats = deduplicate(df)
    print(f"de-duplication: {stats['before']} -> {stats['after']} rows "
          f"({stats['removed']} extra copies removed)")

    train_df, val_df, test_df = split_dataset(df)
    print(f"sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    save_splits(train_df, val_df, test_df)
    print(f"saved train/val/test.csv to {PROCESSED_DIR}")
