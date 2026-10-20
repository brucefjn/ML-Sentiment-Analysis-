# Data provenance and reproducibility

Audit date: October 19, 2026. This is a data and setup audit, not a fresh full-model training run.

## What the input actually contains

The original notebook downloads a CSV from [Google Drive](https://drive.google.com/uc?export=view&id=1kWs6yOYpdjVr-liLPs4PKIs1qSpIohzS). The audit fetched that exact link and found:

| Property | Verified value |
| --- | ---: |
| File size | 31,994,890 bytes |
| Rows | 24,988 |
| Negative / positive | 12,496 / 12,492 |
| Empty reviews or invalid labels | 0 |
| Unique full review texts | 24,892 |
| Extra rows repeating a review text | 96 |
| Repeated text groups / rows within those groups | 92 / 188 |
| Identical texts with conflicting labels | 0 |

The CSV SHA-256 is `6325232cabf13fd098da9a6ae3baa7a34718043c096fc277782651f4c6fd9ff4`. A checksum establishes exact input identity, not ownership or permission to reuse it. Downloaded data and model weights are ignored by Git and are not redistributed here.

## Source identified by exact content matching

Every CSV row, including its label and duplicate multiplicity, matches a transformed subset of the **training split** of [Stanford's Large Movie Review Dataset v1.0](https://ai.stanford.edu/~amaas/data/sentiment/), introduced by [Maas et al. (2011), *Learning Word Vectors for Sentiment Analysis*](https://aclanthology.org/P11-1015/).

The comparison used the official `aclImdb_v1.tar.gz`, downloaded from Stanford: 84,125,825 bytes; SHA-256 `c40f74a18d3b61f90feba1e17730e0d38e8b97c05fde7008942e91923d1658fe`.

For matching only, each original review was transformed in this exact order:

```python
text = text.lower()
text = re.sub(r"[^a-z0-9\s]", "", text)
text = re.sub(r"http\S+|www.\S+", "", text)
```

The `.` after `www` is deliberately shown exactly as used in the successful comparison: it is a regex wildcard, not a literal dot. This can also remove parts of non-URL words. This is **evidence of content correspondence**, not a recovered preprocessing script or a recommended text-cleaning recipe. Whitespace was not collapsed. Negative and positive labels match the source folders, and the comparison accounts for duplicate counts rather than just testing membership in a set.

The CSV omits 12 of the 25,000 official training records: 4 negative and 8 positive. Their source filenames are recorded in [the audit report](reports/data-audit-2026-10-19.json). The identity of the CSV preparer, the reason for these omissions, and the historical preprocessing code remain undocumented. The download URL alone does not resolve those questions.

The archive README requests a citation but does not provide an explicit redistribution license, and a license for this derivative CSV has not been established. This repository supplies code, hashes, counts, and source references rather than a copy of either dataset. It does not assign an invented license to the data.

## Why the saved score needs a qualification

Reconstructing the original stratified 80/20 split (`random_state=20`) with the pinned audit environment yields 19,990 training rows and 4,998 validation rows, matching the saved counts. It also reveals **24 identical review texts on both sides of the split**, affecting 25 validation rows and 25 training rows.

The original notebook's **91.38% accuracy and 91.38% macro-F1** remain preserved as historical outputs. They are not a newly reproduced, duplicate-free benchmark. Their numerical inflation, if any, cannot be measured without retraining and reevaluation; this audit does not invent a corrected score. The validation split was also used for checkpoint selection, so it was never an untouched final test set.

The [corrected rerun notebook](sentiment_prediction_system_clean.ipynb) keeps the original CSV, removes only exact repeated review texts before splitting, and checks that no identical text crosses the split. It produces:

| Split | Negative | Positive | Total |
| --- | ---: | ---: | ---: |
| Training | 9,942 | 9,971 | 19,913 |
| Validation | 2,486 | 2,493 | 4,979 |

This fixes the demonstrated exact-duplicate overlap. It does not claim to eliminate near duplicates, truncation-induced overlap, or movie-level dependence. No official test-set score is claimed. Stanford's official benchmark uses a separate test split and movie-disjoint train/test construction; this course project's random split within the supplied training subset is a different evaluation design.

## Reproduce the lightweight audit

Use Python 3.12 or 3.13. The checked environment was Python 3.13.7 with the versions in `requirements-data.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-data.txt
python scripts/audit_data.py
python -m unittest discover -s tests -v
```

The audit downloads the known CSV if no path is supplied, verifies its SHA-256 before parsing, rejects unexpected columns, blank reviews, invalid labels and contradictory duplicate labels, and reports both the original and corrected split. A changed or HTML download fails clearly rather than becoming training input. If the Drive link stops working, use an already-held copy with the same checksum; do not silently substitute a different dataset.

To repeat the source comparison, download the [official Stanford archive](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz) locally and run:

```bash
python scripts/audit_data.py data/movie_reviews.csv \
  --stanford-archive /path/to/aclImdb_v1.tar.gz \
  --output reports/local-data-audit.json
```

The committed audit report includes ordered split-row hashes: SHA-256 of newline-joined zero-based source row numbers, in split order, without a final newline. It contains no review texts.

## Framework and checkpoint checks

The original saved run identifies Transformers 5.0.0 and logs missing `LayerNorm.weight/bias` keys alongside unexpected `LayerNorm.gamma/beta` keys when loading a selected checkpoint. Its trained checkpoint is not in the repository, so this audit cannot retrospectively verify its weights or assign a new metric.

The new notebook pins direct dependencies in `requirements.txt` and pins the current source model revision in `data-manifest.json`; these are **new-run settings**, not a claim to recover the entire historical environment. Dependencies beneath those direct pins are not fully locked. Install them in a fresh environment before importing the libraries; restart Colab if imports already occurred.

An offline two-step **synthetic tiny-DistilBERT smoke test** passed with Transformers 5.8.0, PyTorch 2.11.0, Accelerate 1.12.0, and Safetensors 0.7.0. It exercised training, best-checkpoint selection, direct reload, export and reload; no missing/unexpected keys were reported and the test logits matched exactly. This is a framework plumbing check, not a trained sentiment model or a reproduction of the 91.38% result.

```bash
python -m pip install -r requirements.txt
python scripts/check_checkpoint.py
```

The corrected notebook checks the selected checkpoint's parameters and export/reload predictions after a future full run. Its training and inference output cells are intentionally empty until that run is performed.
