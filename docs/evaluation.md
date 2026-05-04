# Model Evaluation

## Training configuration

| Parameter     | Value              |
|---------------|--------------------|
| Base model    | `bert-base-uncased`     |
| Train samples | 2000    |
| Epochs        | 3           |
| Max length    | 128 tokens         |
| Batch size    | 16                 |

## Test set results

Evaluated on the held-out test split (2,600 samples, 46.7% positive).

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.8985 |
| Precision | 0.9004 |
| Recall    | 0.8796 |
| F1        | 0.8899 |

### Confusion matrix

|                    | Predicted: Not Claim | Predicted: Claim |
|--------------------|----------------------|------------------|
| **Actual: Not Claim** | 1269 (TN)         | 118 (FP)        |
| **Actual: Claim**     | 146 (FN)         | 1067 (TP)        |

## Interpretation

**Precision (0.9004)** measures how often the model is correct when it predicts
a sentence is a claim. A high precision reduces wasted fact-checking effort on non-claims.

**Recall (0.8796)** measures how many actual claims the model catches. A high recall
ensures few real claims slip through undetected — important for a system meant to feed a
downstream fact-checker.

**F1 (0.8899)** is the harmonic mean of precision and recall. For claim detection,
this is the primary metric of interest since both false positives (wasted fact-checking effort)
and false negatives (missed claims) carry costs.

## Comparison to paper benchmarks

Bell (2025) reports the following results on the same dataset with a full fine-tuning run
(5 epochs, 10,397 training samples):

| Model                | Accuracy | Precision | Recall | F1    |
|----------------------|----------|-----------|--------|-------|
| BERT (Finetuned)     | 0.917    | 0.918     | 0.904  | 0.911 |
| ModernBERT (Finetuned) | 0.911  | 0.907     | 0.902  | 0.904 |
| **This model**       | **0.898** | **0.900** | **0.880** | **0.890** |

Note: this model used a reduced training set (2000 samples, 3 epochs) for
CPU-feasibility. A full training run is expected to achieve results closer to the paper's BERT
benchmark (F1 ~0.911).

## Out-of-domain considerations

Bell (2025) also notes that fine-tuned BERT-based models transfer poorly to out-of-domain data
(e.g., tweets), tending to over-predict claims. This model is trained on political speeches and
fact-checks and should be expected to degrade on domains with different linguistic character
(informal text, social media, etc.). For broad-domain use, an LLM-based approach would be
more robust.
