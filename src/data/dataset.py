import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_SEED, PROCESSED_DIR


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    n_labels_per_clean = df.groupby("clean")["label"].nunique()
    assert (n_labels_per_clean <= 1).all(), "duplicate articles with conflicting labels"
    before = len(df)
    df = df.drop_duplicates(subset="clean", keep="first").reset_index(drop=True)
    print(f"de-duplication: {before} -> {len(df)} rows ({before - len(df)} extra copies removed)")
    return df


def split_dataset(df: pd.DataFrame, random_state: int = RANDOM_SEED):
    train_df, rest_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=random_state
    )
    validation_df, test_df = train_test_split(
        rest_df, test_size=0.50, stratify=rest_df["label"], random_state=random_state
    )
    return train_df, validation_df, test_df


if __name__ == "__main__":
    from src.data.loader import load_dataset
    from src.data.preprocessing import clean_arabic_text

    df = load_dataset()
    df["clean"] = df["text"].apply(clean_arabic_text)
    df = deduplicate(df)

    train_df, val_df, test_df = split_dataset(df)

    print("\n=== check 1: sizes and per-class counts ===")
    counts = pd.DataFrame({
        "train": train_df["label"].value_counts(),
        "validation": val_df["label"].value_counts(),
        "test": test_df["label"].value_counts(),
    })
    print(counts)
    print(f"totals: train={len(train_df)}, validation={len(val_df)}, test={len(test_df)}")

    print("\n=== check 2: no overlap between splits ===")
    for name, a, b in [("train/val", train_df, val_df),
                       ("train/test", train_df, test_df),
                       ("val/test", val_df, test_df)]:
        print(f"overlap {name}: {len(set(a['clean']) & set(b['clean']))}")

    print("\n=== check 3: reproducibility (split twice, same seed) ===")
    train2, val2, test2 = split_dataset(df)
    same = (train_df.index.equals(train2.index)
            and val_df.index.equals(val2.index)
            and test_df.index.equals(test2.index))
    print(f"identical indices on second run: {same}")

    print("\n=== one train article: raw vs clean (first 200 chars) ===")
    row = train_df.iloc[0]
    print("RAW  :", row["text"][:200].replace("\n", " "))
    print("CLEAN:", row["clean"][:200])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
    print(f"\nsaved: {PROCESSED_DIR}/train.csv, val.csv, test.csv")
    print("NOTE: test.csv stays untouched until the final evaluation.")
