# Movie Review Sentiment Classification with DistilBERT

A course project that fine-tunes DistilBERT to classify movie reviews as positive or negative, then explores how the classifier handles new sentences, negation, and sarcasm.

**Author:** Jianing Fan · **Status:** course project; original outputs preserved, corrected data workflow added

[Original notebook](sentiment_prediction_system.ipynb) · [Corrected rerun notebook](sentiment_prediction_system_clean.ipynb) · [Run corrected version in Colab](https://colab.research.google.com/github/brucefjn/ML-Sentiment-Analysis-/blob/main/sentiment_prediction_system_clean.ipynb) · [Portfolio](https://brucefjn.github.io/)

## What I built

- A preprocessing workflow that removes missing reviews and labels, encodes sentiment labels, and creates a stratified 80/20 split.
- A fine-tuning pipeline using Hugging Face Transformers and Datasets, PyTorch, and the `distilbert-base-uncased` checkpoint.
- Evaluation with accuracy and macro-F1, plus sentence-level predictions with class probabilities and an interactive text prompt.
- A small error-analysis exercise examining comparative sentiment, negation, sarcasm, and contrastive phrasing.

Reviews are tokenized to a maximum of 256 tokens. Training uses two epochs, a learning rate of `2e-5`, and batches of 16. The notebook records a Google Colab T4 GPU environment.

## Data verification

The original CSV has now been checked against Stanford's official IMDb v1.0 archive: all 24,988 rows and their labels match a transformed subset of its **training split**. The CSV contains 96 repeated review-text rows. Reconstructing the original split finds 24 identical review texts in both training and validation, affecting 25 validation rows.

The corrected workflow verifies the file checksum, rejects malformed data and conflicting labels, removes exact repeated texts **before splitting**, and checks for overlap. Its prepared split contains 19,913 training and 4,979 validation reviews, with zero identical full texts shared across them. **No new full-model score is claimed.** See [data provenance, the reproducible audit, and remaining limitations](DATA.md).

## Historical saved results

The committed notebook contains a completed training run on 19,990 reviews and evaluation on 4,998 reviews, from 24,988 labeled rows after preprocessing.

| Metric | Saved evaluation result |
| --- | ---: |
| Accuracy | 91.38% |
| Macro-F1 | 91.38% |

These values come from the original notebook's saved outputs and have not been independently reproduced. They include the duplicate overlap identified above and do not describe the corrected split. The evaluation split also guides checkpoint selection, so these are historical validation results rather than an untouched test-set benchmark. The effect of the overlap on the score is unknown.

Saved inference examples include:

| Input | Prediction |
| --- | --- |
| “This movie is not as good.” | Negative |
| “Absolutely fantastic — I loved it.” | Positive |

## Run the corrected notebook

1. Open the corrected Colab link above in a fresh GPU runtime, or clone this repository and open `sentiment_prediction_system_clean.ipynb` from its root.
2. Run the setup and data cells first. They use pinned direct dependencies, the original CSV's checksum and a fixed split seed. The data checks run without a GPU.
3. Complete training and checkpoint validation before running prediction examples. The new notebook has no pre-filled training results; trained weights are not included.

For a lightweight check without full training:

```bash
python -m pip install -r requirements-data.txt
python scripts/audit_data.py
python -m unittest discover -s tests -v
```

The original course notebook and all of its saved output cells remain intact. The [data notes](DATA.md) distinguish verified source correspondence from the CSV's undocumented preparation and licensing history. Neither dataset is copied into this repository.

## Interpretation and next steps

Sentiment classification works well on many direct examples, but the saved stress tests show confidently incorrect predictions for sarcastic reversals such as “Great movie... said no one ever.” Class probabilities should not be interpreted as calibrated certainty.

The original run uses Transformers 5.0.0 and includes checkpoint-loading warnings. The corrected version pins a newer environment and verifies checkpoint selection and save/reload. An offline tiny-model smoke test passed, but that does not verify the absent historical trained checkpoint. A full deduplicated training run and a separately held-out final evaluation remain future work.
