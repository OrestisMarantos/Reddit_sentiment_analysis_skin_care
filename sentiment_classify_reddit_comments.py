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
- If you�re on Windows and Torch install is annoying, you could use:
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


def main():
    # Allow running without CLI flags (e.g., double-click). Defaults point to typical filenames.
    if len(sys.argv) == 1:
        in_path = Path("solid_perfumes_thread.parsed.json")
        out_path = Path("solid_perfumes_thread.sentiment.csv")
        batch_size = 16
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("--input", required=True, help="Path to parsed thread JSON")
        ap.add_argument("--output", required=True, help="Path to output CSV")
        ap.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
        args = ap.parse_args()
        in_path = Path(args.input)
        out_path = Path(args.output)
        batch_size = args.batch_size

    if not in_path.exists():
        print(f"Input JSON not found: {in_path.resolve()}")
        return

    df = load_comments(in_path)
    if df.empty:
        print("No comments found after cleaning. Check your JSON file.")
        return

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
        batch = texts[i : i + batch_size]
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
    df_out.to_csv(out_path, index=False, encoding="utf-8")

    counts = df_out["sentiment"].value_counts()
    print(f"Loaded comments: {len(df_out)}")
    print("Sentiment counts:")
    print(counts.to_string())
    print(f"\nSaved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
