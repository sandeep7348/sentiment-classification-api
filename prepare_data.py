# prepare_data.py — download, merge, and format Kaggle datasets for training

import os
import zipfile
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── Directories ────────────────────────────────────────────────
RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)


# ── Step 1: Download from Kaggle ───────────────────────────────
def download_datasets():
    import subprocess

    datasets = [
        ("kazanova/sentiment140", "data/raw"),

        ("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews", "data/raw"),

        ("snap/amazon-fine-food-reviews", "data/raw"),

        ("crowdflower/twitter-airline-sentiment", "data/raw"),

        ("yelp-dataset/yelp-dataset", "data/raw"),

        ("andrewmvd/steam-reviews", "data/raw"),

        ("nicapotato/womens-ecommerce-clothing-reviews", "data/raw"),

        ("ankurzing/sentiment-analysis-for-financial-news", "data/raw"),

        ("cosmos98/twitter-and-reddit-sentimental-analysis-dataset", "data/raw"),

        ("praveengovi/emotions-dataset-for-nlp", "data/raw"),
    ]

    for dataset, path in datasets:

        name = dataset.split("/")[-1]

        zip_path = Path(path) / f"{name}.zip"

        if zip_path.exists():
            print(f"Already downloaded: {name}")

        else:
            print(f"Downloading {name} ...")

            result = subprocess.run(
                ["kaggle", "datasets", "download", "-d", dataset, "-p", path],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                print(f"  ERROR: {result.stderr.strip()}")

            else:
                print("  Done.")

        # Unzip
        if zip_path.exists():

            print(f"Unzipping {zip_path.name} ...")

            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(path)

            print(f"  Extracted to {path}/")


# ── Step 2: Load & normalise datasets ──────────────────────────

def load_sentiment140() -> pd.DataFrame:

    path = RAW_DIR / "training.1600000.processed.noemoticon.csv"

    if not path.exists():
        print("  [SKIP] Sentiment140 CSV not found.")
        return pd.DataFrame()

    df = pd.read_csv(
        path,
        encoding="latin-1",
        header=None,
        names=["label", "id", "date", "query", "user", "text"]
    )

    df["label"] = df["label"].map({
        0: "negative",
        4: "positive"
    })

    df = df[["text", "label"]].dropna()

    df = df.sample(
        min(50000, len(df)),
        random_state=42
    )

    print(f"  Sentiment140 : {len(df):,} rows")

    return df


def load_imdb() -> pd.DataFrame:

    path = RAW_DIR / "IMDB Dataset.csv"

    if not path.exists():
        print("  [SKIP] IMDb CSV not found.")
        return pd.DataFrame()

    df = pd.read_csv(path)

    df.columns = df.columns.str.strip().str.lower()

    col_map = {}

    for c in df.columns:

        if c in ("review", "text", "comment", "content"):
            col_map[c] = "text"

        if c in ("sentiment", "label", "class", "polarity"):
            col_map[c] = "label"

    df = df.rename(columns=col_map)

    if "text" not in df.columns or "label" not in df.columns:
        print("  [SKIP] IMDb missing required columns.")
        return pd.DataFrame()

    df = df[["text", "label"]].dropna()

    print(f"  IMDb         : {len(df):,} rows")

    return df


def load_amazon() -> pd.DataFrame:

    path = RAW_DIR / "Reviews.csv"

    if not path.exists():
        print("  [SKIP] Amazon Reviews CSV not found.")
        return pd.DataFrame()

    df = pd.read_csv(path, usecols=["Text", "Score"])

    df = df.rename(columns={"Text": "text"})

    df["label"] = df["Score"].map({
        1: "negative",
        2: "negative",
        3: "neutral",
        4: "positive",
        5: "positive",
    })

    df = df[["text", "label"]].dropna()

    df = df.sample(
        min(50000, len(df)),
        random_state=42
    )

    print(f"  Amazon       : {len(df):,} rows")

    return df


def load_airline() -> pd.DataFrame:

    path = RAW_DIR / "Tweets.csv"

    if not path.exists():
        print("  [SKIP] Airline CSV not found.")
        return pd.DataFrame()

    df = pd.read_csv(path)

    df = df.rename(columns={
        "text": "text",
        "airline_sentiment": "label"
    })

    df = df[["text", "label"]].dropna()

    # Oversample neutral class
    neutral = df[df["label"] == "neutral"]

    neutral = neutral.sample(
        min(15000, len(neutral)),
        replace=True,
        random_state=42
    )

    others = df[df["label"] != "neutral"]

    df = pd.concat([others, neutral])

    print(f"  Airline      : {len(df):,} rows")

    return df


# ── Step 3: Merge, clean, balance, save ───────────────────────

def build_dataset():

    print("\nLoading datasets ...")

    parts = [
        load_sentiment140(),
        load_imdb(),
        load_amazon(),
        load_airline(),
    ]

    df = pd.concat(
        [p for p in parts if not p.empty],
        ignore_index=True
    )

    # Basic cleaning
    df["text"] = df["text"].astype(str).str.strip()

    df = df[df["text"].str.len() > 5]

    df = df.drop_duplicates(subset="text")

    # Shuffle
    df = df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    print(f"\nTotal samples : {len(df):,}")

    print("\nLabel distribution:")

    print(df["label"].value_counts().to_string())

    # Save merged file
    out = PROC_DIR / "train.csv"

    df.to_csv(out, index=False)

    print(f"\nSaved → {out}")

    return df


# ── Step 4: Train / Val / Test Split ──────────────────────────

def split_and_save(df: pd.DataFrame):

    X = df["text"].tolist()

    y = df["label"].tolist()

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=0.50,
        random_state=42,
        stratify=y_tmp
    )

    for split, texts, labels in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:

        out = PROC_DIR / f"{split}.csv"

        pd.DataFrame({
            "text": texts,
            "label": labels
        }).to_csv(out, index=False)

        print(f"  {split:5s} → {out} ({len(texts):,} rows)")


# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 50)
    print("Step 1 — Downloading datasets from Kaggle")
    print("=" * 50)

    download_datasets()

    print("\n" + "=" * 50)
    print("Step 2 — Building merged dataset")
    print("=" * 50)

    df = build_dataset()

    print("\n" + "=" * 50)
    print("Step 3 — Splitting into train / val / test")
    print("=" * 50)

    split_and_save(df)

    print("\nAll done! Now run:")

    print("  python train.py --data data/processed/train.csv")
