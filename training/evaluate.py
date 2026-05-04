"""
Evaluates a saved claim detection model on the test set and prints a
metrics report. Also writes docs/evaluation.md with results and interpretation.

Usage:
    python evaluate.py
    python evaluate.py --model-dir training/model_output
"""

import argparse
import json
import os

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_output")
TEST_CSV = os.path.join(REPO_ROOT, "data", "raw", "test.csv")
DOCS_DIR = os.path.join(REPO_ROOT, "docs")
MAX_LENGTH = 128
BATCH_SIZE = 32


def predict(model, tokenizer, texts: list[str], device) -> tuple[list[int], list[float]]:
    model.eval()
    all_preds, all_confs = [], []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits

        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1).cpu().tolist()
        confs = probs.max(dim=-1).values.cpu().tolist()
        all_preds.extend(preds)
        all_confs.extend(confs)

    return all_preds, all_confs


def evaluate(model_dir: str) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)

    print(f"Loading test set from {TEST_CSV}...")
    df = pd.read_csv(TEST_CSV)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    print(f"  {len(texts)} samples")

    print("Running inference...")
    preds, confs = predict(model, tokenizer, texts, device)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)

    results = {
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }

    print("\n--- Results ---")
    for k, v in results.items():
        print(f"  {k}: {v}")

    print("\nClassification report:")
    print(classification_report(labels, preds, target_names=["not_claim", "claim"]))

    print("Confusion matrix:")
    cm = confusion_matrix(labels, preds)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    # Load training metadata if available
    meta = {}
    metrics_path = os.path.join(model_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            meta = json.load(f)

    write_evaluation_doc(results, cm, meta)
    return results


def write_evaluation_doc(results: dict, cm, meta: dict) -> None:
    os.makedirs(DOCS_DIR, exist_ok=True)
    train_samples = meta.get("train_samples", "unknown")
    epochs = meta.get("epochs", "unknown")
    model_name = meta.get("model", "bert-base-uncased")

    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    content = f"""# Model Evaluation

## Training configuration

| Parameter     | Value              |
|---------------|--------------------|
| Base model    | `{model_name}`     |
| Train samples | {train_samples}    |
| Epochs        | {epochs}           |
| Max length    | 128 tokens         |
| Batch size    | 16                 |

## Test set results

Evaluated on the held-out test split (2,600 samples, 46.7% positive).

| Metric    | Score  |
|-----------|--------|
| Accuracy  | {results['accuracy']:.4f} |
| Precision | {results['precision']:.4f} |
| Recall    | {results['recall']:.4f} |
| F1        | {results['f1']:.4f} |

### Confusion matrix

|                    | Predicted: Not Claim | Predicted: Claim |
|--------------------|----------------------|------------------|
| **Actual: Not Claim** | {tn} (TN)         | {fp} (FP)        |
| **Actual: Claim**     | {fn} (FN)         | {tp} (TP)        |

## Interpretation

**Precision ({results['precision']:.4f})** measures how often the model is correct when it predicts
a sentence is a claim. A high precision reduces wasted fact-checking effort on non-claims.

**Recall ({results['recall']:.4f})** measures how many actual claims the model catches. A high recall
ensures few real claims slip through undetected — important for a system meant to feed a
downstream fact-checker.

**F1 ({results['f1']:.4f})** is the harmonic mean of precision and recall. For claim detection,
this is the primary metric of interest since both false positives (wasted fact-checking effort)
and false negatives (missed claims) carry costs.

## Comparison to paper benchmarks

Bell (2025) reports the following results on the same dataset with a full fine-tuning run
(5 epochs, 10,397 training samples):

| Model                | Accuracy | Precision | Recall | F1    |
|----------------------|----------|-----------|--------|-------|
| BERT (Finetuned)     | 0.917    | 0.918     | 0.904  | 0.911 |
| ModernBERT (Finetuned) | 0.911  | 0.907     | 0.902  | 0.904 |
| **This model**       | **{results['accuracy']:.3f}** | **{results['precision']:.3f}** | **{results['recall']:.3f}** | **{results['f1']:.3f}** |

Note: this model used a reduced training set ({train_samples} samples, {epochs} epochs) for
CPU-feasibility. A full training run is expected to achieve results closer to the paper's BERT
benchmark (F1 ~0.911).

## Out-of-domain considerations

Bell (2025) also notes that fine-tuned BERT-based models transfer poorly to out-of-domain data
(e.g., tweets), tending to over-predict claims. This model is trained on political speeches and
fact-checks and should be expected to degrade on domains with different linguistic character
(informal text, social media, etc.). For broad-domain use, an LLM-based approach would be
more robust.
"""

    out_path = os.path.join(DOCS_DIR, "evaluation.md")
    with open(out_path, "w") as f:
        f.write(content)
    print(f"\nEvaluation report written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate claim detection model")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Path to saved model directory")
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        print(f"Model not found at {args.model_dir}. Run train.py first.")
        raise SystemExit(1)

    evaluate(args.model_dir)
