#!/usr/bin/env python3
"""Generic FinBERT sentiment runner.

Loads ProsusAI/finbert via a transformers pipeline on CPU and scores an
arbitrary list of English financial texts (any market, any source).

Usage:
    python scripts/finbert_sentiment.py --input texts.txt [--label my-run]
    python scripts/finbert_sentiment.py --input texts.json [--label my-run]

Input formats:
  - .txt: one text per line (blank lines skipped)
  - .json: a JSON list of strings (or objects with a "text" field)

Outputs (into results/):
  - sentiment.json: per-item label + probabilities, label counts, mean probs
  - report.md: human-readable table of the same
"""
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
from collections import Counter
from pathlib import Path

MODEL_ID = "ProsusAI/finbert"
LABELS = ("positive", "negative", "neutral")


def load_texts(path: Path):
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        texts = []
        for item in data:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and "text" in item:
                texts.append(str(item["text"]))
            else:
                raise ValueError(f"Unsupported JSON item: {item!r}")
        return [t for t in texts if t.strip()]
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run FinBERT sentiment over a list of texts.")
    parser.add_argument("--input", required=True, help="Path to a .txt (one text per line) or .json (list) file")
    parser.add_argument("--label", default="run", help="Label for this run (recorded in the outputs)")
    args = parser.parse_args()

    from transformers import pipeline

    texts = load_texts(Path(args.input))
    if not texts:
        raise SystemExit("No texts found in input.")

    clf = pipeline("sentiment-analysis", model=MODEL_ID, top_k=None, device=-1)
    raw_outputs = clf(texts)

    items = []
    for text, scores in zip(texts, raw_outputs):
        probs = {label: 0.0 for label in LABELS}
        for entry in scores:
            probs[entry["label"].lower()] = float(entry["score"])
        predicted = max(probs, key=probs.get)
        items.append({"text": text, "predicted": predicted, "probabilities": probs})

    counts = Counter(item["predicted"] for item in items)
    means = {label: sum(item["probabilities"][label] for item in items) / len(items) for label in LABELS}

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "label": args.label,
        "model": MODEL_ID,
        "count": len(items),
        "label_counts": {label: counts.get(label, 0) for label in LABELS},
        "mean_probabilities": means,
        "items": items,
    }
    (results_dir / "sentiment.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"# FinBERT sentiment — {args.label}",
        "",
        f"- Model: `{MODEL_ID}` (CPU)",
        f"- Texts scored: {len(items)}",
        f"- Label counts: " + ", ".join(f"{label}={counts.get(label, 0)}" for label in LABELS),
        f"- Mean probabilities: " + ", ".join(f"{label}={means[label]:.4f}" for label in LABELS),
        "",
        "| # | Predicted | P(pos) | P(neg) | P(neu) | Text |",
        "|---|-----------|--------|--------|--------|------|",
    ]
    for i, item in enumerate(items, 1):
        p = item["probabilities"]
        text = item["text"].replace("|", "\\|")
        lines.append(f"| {i} | {item['predicted']} | {p['positive']:.4f} | {p['negative']:.4f} | {p['neutral']:.4f} | {text} |")
    lines.append("")
    (results_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Scored {len(items)} texts | counts={dict(counts)} | means=" +
          ", ".join(f"{label}={means[label]:.4f}" for label in LABELS))


if __name__ == "__main__":
    main()
