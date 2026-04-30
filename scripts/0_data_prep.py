"""
Data Preparation: Load a single parquet file, clean, and split into
train (70%) / eval (15%) / test (15%).
"""

import pandas as pd
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_FILE = DATA_DIR / "train-00000-of-00003.parquet"

CODE_COL = "code"
COMMENT_COL = "comment"

CODE_MAX_WORDS = 400
COMMENT_MAX_WORDS = 100
COMMENT_MIN_WORDS = 3

TRAIN_FRAC = 0.70
EVAL_FRAC  = 0.15
# test gets the remaining 0.15


def load(path: Path) -> pd.DataFrame:
    logger.info(f"Loading {path.name} …")
    df = pd.read_parquet(path, columns=[CODE_COL, COMMENT_COL])
    logger.info(f"  {len(df):,} rows")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=[CODE_COL, COMMENT_COL])
    df = df[(df[CODE_COL].str.strip() != "") & (df[COMMENT_COL].str.strip() != "")]

    code_words    = df[CODE_COL].str.split().str.len()
    comment_words = df[COMMENT_COL].str.split().str.len()
    df = df[
        (code_words    <= CODE_MAX_WORDS)
        & (comment_words >= COMMENT_MIN_WORDS)
        & (comment_words <= COMMENT_MAX_WORDS)
    ]
    logger.info(f"Cleaned: {before:,} → {len(df):,} rows")
    return df[[CODE_COL, COMMENT_COL]].copy()


def split(df: pd.DataFrame, seed: int = 42):
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train = int(n * TRAIN_FRAC)
    n_eval  = int(n * EVAL_FRAC)

    train_df = df.iloc[:n_train].copy()
    eval_df  = df.iloc[n_train : n_train + n_eval].copy()
    test_df  = df.iloc[n_train + n_eval :].copy()

    logger.info(f"Split → train: {len(train_df):,}  eval: {len(eval_df):,}  test: {len(test_df):,}")
    return train_df, eval_df, test_df


def save_stats(train_df, eval_df, test_df, total_before_clean: int):
    stats = {
        "source_file": SOURCE_FILE.name,
        "rows_before_clean": total_before_clean,
        "rows_after_clean": len(train_df) + len(eval_df) + len(test_df),
        "train_rows": len(train_df),
        "eval_rows":  len(eval_df),
        "test_rows":  len(test_df),
        "avg_code_words_train":    round(train_df[CODE_COL].str.split().str.len().mean(), 1),
        "avg_comment_words_train": round(train_df[COMMENT_COL].str.split().str.len().mean(), 1),
    }
    out = PROCESSED_DIR / "data_stats.json"
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats:\n{json.dumps(stats, indent=2)}")


def run():
    logger.info("=" * 60)
    logger.info("DATA PREPARATION PIPELINE")
    logger.info("=" * 60)

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Source file not found: {SOURCE_FILE}")

    df = load(SOURCE_FILE)
    raw_count = len(df)

    df = clean(df)

    train_df, eval_df, test_df = split(df)

    train_df.to_parquet(PROCESSED_DIR / "train_split.parquet", index=False)
    eval_df.to_parquet(PROCESSED_DIR  / "eval_split.parquet",  index=False)
    test_df.to_parquet(PROCESSED_DIR  / "test_split.parquet",  index=False)

    save_stats(train_df, eval_df, test_df, raw_count)
    logger.info("Data preparation complete.")
    return train_df, eval_df, test_df


if __name__ == "__main__":
    train_df, eval_df, test_df = run()
    print(f"\n✓ Data prepared — saved to {PROCESSED_DIR}/")
    print(f"  Train : {len(train_df):,}")
    print(f"  Eval  : {len(eval_df):,}")
    print(f"  Test  : {len(test_df):,}")
