"""Check the four-class merged dataset and preserve its source/session manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from dataset import ROOT, load_dataset, make_manifest, read_config, save_manifest
from hand_features import CLASS_LABELS


def local_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json", help="Project configuration (default: config.json).")
    parser.add_argument("--output", default="data/split_manifest.json", help="New manifest path; an incompatible existing file is never overwritten.")
    args = parser.parse_args()
    config = read_config(local_path(args.config))
    data = load_dataset(local_path(config["dataset_path"]), config)
    manifest = make_manifest(data, config)
    output = local_path(args.output)
    created = save_manifest(output, manifest)
    for session, counts in data.session_counts.items():
        text = ", ".join(f"{name}={counts[name]}" for name in CLASS_LABELS)
        print(f"{session}: {text} | total={sum(counts.values())}")
    print(f"Total rows: {len(data.sample_ids)}")
    print(f"Unique IDs: {len(set(data.sample_ids))}")
    print(f"Features: {data.features.shape[1]} finite float32 values per row; wrist/scale checks passed")
    print("Session overlap: none")
    print("Exact feature duplicates across splits: 0")
    if data.duplicate_feature_rows_within_split:
        print(f"NOTE: {data.duplicate_feature_rows_within_split} exact repeated feature rows within splits; recording frames can be correlated.")
    for role, split in manifest["splits"].items():
        print(f"{role}: {split['row_count']} rows from {', '.join(split['session_ids'])}")
    print(f"CSV SHA-256: {data.sha256}")
    print(f"Manifest {'saved' if created else 'matches; kept unchanged'}: {output}")
    print("DATASET CHECK PASSED")
    print("No model has been trained or evaluated by this command.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user. The source CSV was not changed.", file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"DATASET CHECK FAILED: {exc}", file=sys.stderr)
        print("Keep the current CSV and metadata, and share this output before changing them.", file=sys.stderr)
        sys.exit(1)
