"""Validate the original course CSV and prepare a duplicate-free validation split."""

import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent


def manifest():
    return json.loads((ROOT / "data-manifest.json").read_text())


def validate_csv(path, expected_sha256=None):
    """Reject substituted downloads, malformed rows, blanks and unknown labels."""
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Dataset checksum mismatch: {digest}. Do not train on this file.")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
    if reader.fieldnames != ["Review", "Emotion"]:
        raise ValueError("Expected exactly the CSV columns Review,Emotion.")
    rows = list(reader)
    if not rows:
        raise ValueError("Dataset is empty.")
    seen = {}
    for line, row in enumerate(rows, 2):
        if None in row or not isinstance(row["Review"], str) or not row["Review"].strip():
            raise ValueError(f"Missing review or malformed CSV at line {line}.")
        if row["Emotion"] not in {"negative", "positive"}:
            raise ValueError(f"Invalid sentiment label at line {line}.")
        if row["Review"] in seen and seen[row["Review"]] != row["Emotion"]:
            raise ValueError(f"Conflicting labels for identical review text at line {line}.")
        seen[row["Review"]] = row["Emotion"]
    counts = Counter(row["Review"] for row in rows)
    return {
        "sha256": digest,
        "bytes": len(raw),
        "rows": len(rows),
        "class_counts": dict(sorted(Counter(row["Emotion"] for row in rows).items())),
        "unique_review_texts": len(counts),
        "duplicate_text_groups": sum(n > 1 for n in counts.values()),
        "extra_duplicate_rows": len(rows) - len(counts),
        "rows_in_duplicate_groups": sum(n for n in counts.values() if n > 1),
        "conflicting_label_groups": 0,
    }


def download_original(path=ROOT / "data/movie_reviews.csv"):
    """Download only the known input, validating before installing it at path."""
    path = Path(path)
    source = manifest()["csv"]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".download")
        try:
            with urlopen(source["url"], timeout=90) as response:
                temporary.write_bytes(response.read())
            validate_csv(temporary, source["sha256"])
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    validate_csv(path, source["sha256"])
    return path


def prepare_splits(path, deduplicate=True):
    """Use the historical seed; deduplicate exact text before splitting by default."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    report = validate_csv(path, manifest()["csv"]["sha256"])
    frame = pd.read_csv(path, keep_default_na=False)
    frame["source_row"] = range(len(frame))
    frame["label"] = frame["Emotion"].map({"negative": 0, "positive": 1})
    if deduplicate:
        frame = frame.drop_duplicates("Review", keep="first").copy()
    train, validation = train_test_split(
        frame, test_size=0.2, random_state=20, stratify=frame["label"]
    )
    shared = set(train.Review) & set(validation.Review)
    if deduplicate and shared:
        raise AssertionError("Identical review text crossed the split.")

    def row_hash(part):
        encoded = "\n".join(str(int(i)) for i in part.source_row).encode()
        return hashlib.sha256(encoded).hexdigest()

    report["split"] = {
        "deduplicated": deduplicate,
        "random_state": 20,
        "validation_fraction": 0.2,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_class_counts": train.Emotion.value_counts().sort_index().to_dict(),
        "validation_class_counts": validation.Emotion.value_counts().sort_index().to_dict(),
        "shared_review_texts": len(shared),
        "training_rows_with_shared_text": int(train.Review.isin(shared).sum()),
        "validation_rows_with_shared_text": int(validation.Review.isin(shared).sum()),
        "ordered_train_source_rows_sha256": row_hash(train),
        "ordered_validation_source_rows_sha256": row_hash(validation),
    }
    return train, validation, report
