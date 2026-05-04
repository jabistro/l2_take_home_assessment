"""
Fine-tunes bert-base-uncased for binary sentence-level claim detection.

Default (demo) run: 2,000 training samples, 3 epochs (~20-30 min on CPU).
Full run:          all 10,397 training samples, 5 epochs (~8-15h on CPU).

Usage:
    python train.py              # demo run
    python train.py --full       # full run
    python train.py --max-samples 500 --epochs 2  # custom
"""

import argparse
import os
import json

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "raw")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "model_output")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

MODEL_NAME = "bert-base-uncased"
DEMO_MAX_SAMPLES = 2000
DEMO_EPOCHS = 3
FULL_EPOCHS = 5
MAX_LENGTH = 128


class ClaimDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_split(path: str, max_samples: int | None = None) -> tuple[list[str], list[int]]:
    df = pd.read_csv(path)
    if max_samples is not None:
        df = df.sample(n=min(max_samples, len(df)), random_state=42)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    return texts, labels


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def train(max_samples: int | None, epochs: int) -> None:
    print(f"\nLoading dataset (max_samples={max_samples or 'all'})...")
    train_texts, train_labels = load_split(os.path.join(DATA_DIR, "train.csv"), max_samples)
    test_texts, test_labels = load_split(os.path.join(DATA_DIR, "test.csv"))
    print(f"  Train: {len(train_texts)} samples | Test: {len(test_texts)} samples")

    print(f"\nLoading tokenizer ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    test_enc = tokenizer(test_texts, truncation=True, padding=True, max_length=MAX_LENGTH)

    train_dataset = ClaimDataset(train_enc, train_labels)
    test_dataset = ClaimDataset(test_enc, test_labels)

    print(f"\nLoading model ({MODEL_NAME})...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=os.path.join(CHECKPOINT_DIR, "logs"),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print(f"\nFine-tuning for {epochs} epoch(s)...")
    trainer.train()

    print("\nRunning final evaluation on test set...")
    metrics = trainer.evaluate()
    print(f"\nTest set results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "model": MODEL_NAME,
                "epochs": epochs,
                "train_samples": len(train_texts),
                "test_samples": len(test_texts),
                "metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
            },
            f,
            indent=2,
        )

    print(f"\nModel saved to {OUTPUT_DIR}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BERT for claim detection")
    parser.add_argument("--full", action="store_true", help="Run full training (10k+ samples, 5 epochs)")
    parser.add_argument("--max-samples", type=int, default=None, help="Override max training samples")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    args = parser.parse_args()

    if args.full:
        max_samples = None
        epochs = FULL_EPOCHS
    else:
        max_samples = args.max_samples if args.max_samples is not None else DEMO_MAX_SAMPLES
        epochs = args.epochs if args.epochs is not None else DEMO_EPOCHS

    train(max_samples=max_samples, epochs=epochs)
