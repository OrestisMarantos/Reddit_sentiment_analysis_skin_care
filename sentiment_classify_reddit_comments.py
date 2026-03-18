"""
sentiment_classify_reddit_comments.py

Reads your parsed Reddit-thread JSON and classifies each comment as:
  - positive
  - neutral
  - negative

It uses a pretrained transformer sentiment model (AI):
  cardiffnlp/twitter-roberta-base-sentiment-latest
which outputs POSITIVE / NEUTRAL / NEGATIVE.

USAGE:
  pip install -U transformers torch pandas

  python sentiment_classify_reddit_comments.py \
      --input solid_perfumes_thread.parsed.json \
      --output solid_perfumes_thread.sentiment.csv

NOTES:
- The first run will download the model once (then it will be cached locally).
- If you are on Windows and Torch install is annoying, you could use:
    pip install -U transformers torch --index-url https://download.pytorch.org/whl/cpu
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from transformers import pipeline


AVATAR_LINE_RE = re.compile(r"\bu\/[A-Za-z0-9_\-]+\s+avatar\b", re.IGNORECASE)


def clean_comment_text(text: str) -> str:
    """Remove parsing artifacts and normalize whitespace."""
    if not isinstance(text, str):
        return ""
    text = AVATAR_LINE_RE.sub("", text)
    text = text.replace("•\n", "").replace("\n•\n", "\n")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_comments(json_path: Path) -> pd.DataFrame:
    data = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
    comments = data.get("comments", [])
    df = pd.DataFrame(comments)

    for col in ["author", "age", "text", "upvotes", "downvotes"]:
        if col not in df.columns:
            df[col] = None

    df["text_clean"] = df["text"].apply(clean_comment_text)
    df = df[df["text_clean"].str.len() > 0].reset_index(drop=True)
    return df


def classify_json_to_csv(json_path: str | Path, output_csv_path: str | Path, clf=None, batch_size: int = 16) -> Path:
    json_path = Path(json_path)
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_comments(json_path)
    if df.empty:
        raise ValueError(f"No comments found after cleaning in {json_path.name}")

    if clf is None:
        clf = pipeline(
            task="sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
            truncation=True,
            max_length=256,
        )

    texts = df["text_clean"].tolist()
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        results.extend(clf(batch))

    df["sentiment"] = [r["label"].lower() for r in results]
    df["sentiment_score"] = [float(r["score"]) for r in results]

    keep_cols = [
        "author",
        "age",
        "upvotes",
        "downvotes",
        "sentiment",
        "sentiment_score",
        "text_clean",
    ]
    df_out = df[keep_cols].copy()
    df_out.to_csv(output_csv_path, index=False, encoding="utf-8")

    return output_csv_path


def classify_json_files_to_csvs(json_files, output_dir: str | Path, batch_size: int = 16):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clf = pipeline(
        task="sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
        truncation=True,
        max_length=256,
    )

    csv_paths = []
    for json_file in json_files:
        json_file = Path(json_file)
        out_csv = output_dir / f"{json_file.stem.replace('.parsed', '')}.sentiment.csv"
        csv_path = classify_json_to_csv(json_file, out_csv, clf=clf, batch_size=batch_size)
        csv_paths.append(csv_path)

    return csv_paths


def combine_csv_files(csv_files, combined_csv_path: str | Path) -> Path:
    combined_csv_path = Path(combined_csv_path)
    combined_csv_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df["source_csv"] = Path(csv_file).name
        frames.append(df)

    if not frames:
        raise ValueError("No CSV files to combine.")

    combined_df = pd.concat(frames, ignore_index=True)
    combined_df.to_csv(combined_csv_path, index=False, encoding="utf-8")
    return combined_csv_path


# def main():
#     # Allow running without CLI flags (e.g., double-click). Defaults point to typical filenames.
#     if len(sys.argv) == 1:
#         in_path = Path("solid_perfumes_thread.parsed.json")
#         out_path = Path("solid_perfumes_thread.sentiment.csv")
#         batch_size = 16
#     else:
#         ap = argparse.ArgumentParser()
#         ap.add_argument("--input", required=True, help="Path to parsed thread JSON")
#         ap.add_argument("--output", required=True, help="Path to output CSV")
#         ap.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
#         args = ap.parse_args()
#         in_path = Path(args.input)
#         out_path = Path(args.output)
#         batch_size = args.batch_size

#     if not in_path.exists():
#         print(f"Input JSON not found: {in_path.resolve()}")
#         return

#     df = load_comments(in_path)
#     if df.empty:
#         print("No comments found after cleaning. Check your JSON file.")
#         return

#     clf = pipeline(
#         task="sentiment-analysis",
#         model="cardiffnlp/twitter-roberta-base-sentiment-latest",
#         tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
#         truncation=True,
#         max_length=256,
#     )

#     texts = df["text_clean"].tolist()

#     results = []
#     for i in range(0, len(texts), batch_size):
#         batch = texts[i : i + batch_size]
#         results.extend(clf(batch))

#     df["sentiment"] = [r["label"].lower() for r in results]
#     df["sentiment_score"] = [float(r["score"]) for r in results]

#     keep_cols = [
#         "author",
#         "age",
#         "upvotes",
#         "downvotes",
#         "sentiment",
#         "sentiment_score",
#         "text_clean",
#     ]
#     df_out = df[keep_cols].copy()
#     df_out.to_csv(out_path, index=False, encoding="utf-8")

#     counts = df_out["sentiment"].value_counts()
#     print(f"Loaded comments: {len(df_out)}")
#     print("Sentiment counts:")
#     print(counts.to_string())
#     print(f"\nSaved: {out_path.resolve()}")


# if __name__ == "__main__":
#     main()
