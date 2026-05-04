"""
Loads the fine-tuned BERT model and runs inference.
Singleton pattern so the model is loaded once at startup.
"""

import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_LENGTH = 128
_tokenizer = None
_model = None
_device = None


def _get_model_dir() -> str:
    env = os.environ.get("MODEL_DIR")
    if env:
        return env
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(repo_root, "training", "model_output")


def load_model() -> None:
    global _tokenizer, _model, _device

    model_dir = _get_model_dir()
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Model not found at {model_dir}. "
            "Run 'python training/train.py' first, or set MODEL_DIR env var."
        )

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = AutoTokenizer.from_pretrained(model_dir)
    _model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(_device)
    _model.eval()


def is_loaded() -> bool:
    return _model is not None


def predict(sentence: str) -> tuple[bool, float]:
    if not is_loaded():
        raise RuntimeError("Model not loaded. Call load_model() first.")

    enc = _tokenizer(
        sentence,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    enc = {k: v.to(_device) for k, v in enc.items()}

    with torch.no_grad():
        logits = _model(**enc).logits

    probs = torch.softmax(logits, dim=-1).squeeze()
    pred_class = int(probs.argmax().item())
    confidence = float(probs[pred_class].item())

    return bool(pred_class == 1), round(confidence, 4)
