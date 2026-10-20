"""Audit checksums, duplicate leakage, and optional original Stanford provenance."""

import argparse
import collections
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_utils import download_original, manifest, prepare_splits


def matching_transform(text):
    # A verified text-matching transform, not a claim about the CSV author's code.
    return re.sub(r"http\S+|www.\S+", "", re.sub(r"[^a-z0-9\s]", "", text.lower()))


def match_stanford(csv_frame, archive):
    digest = hashlib.sha256(Path(archive).read_bytes()).hexdigest()
    if digest != manifest()["stanford_archive"]["sha256"]:
        raise ValueError("Stanford archive checksum mismatch.")
    original = collections.Counter()
    members = collections.defaultdict(list)
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            if not re.fullmatch(r"aclImdb/train/(pos|neg)/\d+_\d+\.txt", member.name):
                continue
            raw = handle.extractfile(member).read().decode("utf-8")
            label = {"pos": "positive", "neg": "negative"}[member.name.split("/")[2]]
            key = matching_transform(raw), label
            original[key] += 1
            members[key].append(member.name)
    supplied = collections.Counter(zip(csv_frame.Review, csv_frame.Emotion))
    missing = original - supplied
    return {
        "archive_sha256": digest,
        "stanford_train_rows": sum(original.values()),
        "csv_rows_not_matched_in_train_with_same_label": sum((supplied - original).values()),
        "train_rows_not_represented_in_csv": sum(missing.values()),
        "omitted_source_files": sorted(name for key in missing for name in members[key]),
        "note": "Exact multiset match after matching_transform, including labels; original CSV preparation history remains unknown.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", help="Original CSV (download if omitted)")
    parser.add_argument("--stanford-archive", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    path = Path(args.csv) if args.csv else download_original()
    train, validation, original = prepare_splits(path, deduplicate=False)
    _, _, clean = prepare_splits(path)
    report = {
        "scope": "Data audit only; no model training or new accuracy estimate.",
        "python": sys.version.split()[0],
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn")},
        "original": original,
        "deduplicated": clean,
    }
    if args.stanford_archive:
        import pandas as pd
        frame = pd.concat([train, validation]).sort_values("source_row")
        report["stanford_match"] = match_stanford(frame, args.stanford_archive)
    result = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
    print(result)


if __name__ == "__main__":
    main()
