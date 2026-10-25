"""Read validated, already-normalized features and preserve session splits.

This module does not open a camera, normalize saved features again, fit a
model, or choose settings from held-out labels. It is reusable by training.
"""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from hand_features import CLASS_LABELS, FEATURE_COUNT, FEATURE_VERSION, GESTURE_SET_VERSION, GESTURE_POSES
from palm_features import MODEL_INPUT_VERSION

ROOT = Path(__file__).resolve().parent
FEATURE_COLUMNS = [f"f{i:02d}" for i in range(FEATURE_COUNT)]
CSV_COLUMNS = ["sample_id", "session_id", "label_id", "timestamp_utc", "feature_version", *FEATURE_COLUMNS]
EXPECTED_SPLITS = {"train": ["s01", "s02", "s03", "idx_train01", "idx_train02", "idx_train03"],
                   "validation": ["s04", "idx_val01"], "test": ["simple_test01"]}
SESSION_LABELS = {**{s: (0, 2, 3) for s in ("s01", "s02", "s03", "s04")},
                  **{s: (1,) for s in ("idx_train01", "idx_train02", "idx_train03", "idx_val01")},
                  "simple_test01": (0, 1, 2, 3)}
SESSION_SOURCE = {s: ("baseline" if s.startswith("s0") else "fresh_test" if s == "simple_test01" else "index_up")
                  for s in SESSION_LABELS}


def row_fingerprint(row):
    """Hash original textual cells; migration preserves sample/session/feature values."""
    values = [row[column] for column in CSV_COLUMNS]
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_config(path: Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("config.json must contain a JSON object")
    if config.get("model_input_version") != MODEL_INPUT_VERSION:
        raise ValueError("Configuration must select palm_frame_v1 model inputs")
    if config.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Configuration feature version is incompatible")
    if config.get("class_labels") != list(CLASS_LABELS):
        raise ValueError("Configuration class order must be neutral,next,previous,blackout")
    if config.get("gesture_set_version") != GESTURE_SET_VERSION or config.get("gesture_poses") != GESTURE_POSES:
        raise ValueError("Configuration must use the simple_index_v2 INDEX-UP gesture vocabulary")
    if config.get("dataset_path") != "data/samples.csv":
        raise ValueError("Keep the portable four-class dataset_path data/samples.csv")
    if config.get("working_hand") not in ("right", "left"):
        raise ValueError("Set working_hand to the confirmed physical hand in config.json")
    if config.get("input_orientation") != "unflipped_camera":
        raise ValueError("This representation uses unflipped camera input")
    if type(config.get("camera_index")) is not int or config["camera_index"] < 0:
        raise ValueError("Configuration camera_index must be a non-negative integer")
    validate_split_sessions(config.get("split_sessions"))
    return config


def validate_split_sessions(splits) -> dict[str, str]:
    if not isinstance(splits, dict) or set(splits) != set(EXPECTED_SPLITS):
        raise ValueError("Specify train, validation, and test session lists")
    session_to_split = {}
    for role in EXPECTED_SPLITS:
        sessions = splits[role]
        if not isinstance(sessions, list) or not sessions or not all(isinstance(s, str) and s for s in sessions):
            raise ValueError(f"{role} requires a nonempty list of session IDs")
        for session in sessions:
            if session in session_to_split:
                raise ValueError(f"Session overlap or duplicate session: {session}")
            session_to_split[session] = role
    if splits != EXPECTED_SPLITS:
        raise ValueError("Keep the declared simple_v2 session split; do not move validation/test sessions")
    return session_to_split


@dataclass
class ValidatedDataset:
    path: Path
    sha256: str
    metadata_sha256: str
    metadata: dict
    features: np.ndarray
    labels: np.ndarray
    sample_ids: list[str]
    session_ids: list[str]
    split_indices: dict[str, np.ndarray]
    session_counts: dict[str, dict[str, int]]
    duplicate_feature_rows_within_split: int
    expected_per_class: int


def load_dataset(csv_path: Path, config: dict, *, expected_per_class: int = 60) -> ValidatedDataset:
    """Validate all rows without modifying data or re-normalizing its features."""
    if type(expected_per_class) is not int or expected_per_class <= 0:
        raise ValueError("Expected samples per class must be a positive integer")
    session_to_split = validate_split_sessions(config.get("split_sessions"))
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Finish collecting data/samples.csv first.")
    if path.name == "debug_samples.csv":
        raise ValueError("Debug recordings cannot be used as the real dataset")
    meta_path = path.with_suffix(".metadata.json")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}; the dataset's feature/hand metadata is required.")
    metadata = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    expected_metadata = {
        "feature_version": FEATURE_VERSION,
        "class_labels": list(CLASS_LABELS),
        "gesture_set_version": GESTURE_SET_VERSION, "gesture_poses": GESTURE_POSES,
        "physical_hand": config.get("working_hand"),
        "camera_index": config.get("camera_index"),
        "input_orientation": "unflipped_camera",
    }
    if not isinstance(metadata, dict) or any(metadata.get(k) != v for k, v in expected_metadata.items()):
        raise ValueError("Dataset metadata and configuration disagree about features, classes, hand, camera, or orientation")
    lineage_path = path.with_name("lineage.json")
    if not lineage_path.is_file() or file_sha256(lineage_path) != metadata.get("lineage_sha256"):
        raise ValueError("Missing or changed lineage.json; preserve files and rerun the preparation check")
    lineage_hash = file_sha256(lineage_path)
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    origins = lineage.get("samples") if isinstance(lineage, dict) else None
    if (not isinstance(lineage, dict) or lineage.get("gesture_set_version") != GESTURE_SET_VERSION
            or not isinstance(origins, dict) or lineage.get("raw_sources") != metadata.get("raw_sources")):
        raise ValueError("Invalid four-class source lineage")
    original_hash = file_sha256(path)
    meta_hash = file_sha256(meta_path)
    features, labels, sample_ids, session_ids = [], [], [], []
    seen_ids = set()
    seen_vectors = {}
    duplicate_vectors = 0
    counts = Counter()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError("CSV columns must match the collector's metadata + f00 through f62 schema")
        for line_number, row in enumerate(reader, start=2):
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("wrong number of CSV fields")
                sample_id = row["sample_id"]
                if not sample_id or sample_id != sample_id.strip() or sample_id in seen_ids:
                    raise ValueError("missing/invalid or duplicate sample ID")
                session = row["session_id"]
                if session not in session_to_split:
                    raise ValueError(f"unexpected simple_v2 session {session!r}")
                label = int(row["label_id"])
                if label not in range(len(CLASS_LABELS)):
                    raise ValueError("label ID must be 0-3")
                if label not in SESSION_LABELS[session]:
                    raise ValueError("Unexpected label/session pairing; old thumbs-up and OK rows must not enter simple_v2")
                origin = origins.get(sample_id)
                if (not isinstance(origin, dict) or origin.get("source_key") != SESSION_SOURCE[session]
                        or origin.get("row_sha256") != row_fingerprint(row)):
                    raise ValueError("Sample source lineage does not match the preserved row")
                if row["feature_version"] != FEATURE_VERSION:
                    raise ValueError("incompatible feature version")
                recorded_at = datetime.fromisoformat(row["timestamp_utc"])
                if recorded_at.utcoffset() is None or recorded_at.utcoffset().total_seconds() != 0:
                    raise ValueError("timestamp must be a timezone-aware UTC timestamp")
                vector64 = np.asarray([row[column] for column in FEATURE_COLUMNS], dtype=np.float64)
                if not np.isfinite(vector64).all():
                    raise ValueError("non-finite feature value")
                with np.errstate(over="ignore", invalid="ignore"):
                    vector = vector64.astype(np.float32)
                if vector.shape != (FEATURE_COUNT,) or not np.isfinite(vector).all():
                    raise ValueError("features must be 63 finite float32 values")
                if not np.allclose(vector[:3], 0, rtol=0, atol=1e-6):
                    raise ValueError("wrist is not centered; check shared preprocessing")
                if not np.isclose(np.linalg.norm(vector[27:30]), 1, rtol=1e-4, atol=1e-5):
                    raise ValueError("palm scale is not normalized; do not normalize saved rows again")
                role = session_to_split[session]
                key = vector.tobytes()
                earlier = seen_vectors.get(key)
                if earlier is not None:
                    old_role, old_label, old_line = earlier
                    if old_role != role:
                        raise ValueError(f"identical feature vector appears across splits at rows {old_line} and {line_number}; investigate possible copied data")
                    if old_label != label:
                        raise ValueError(f"identical features have conflicting labels at rows {old_line} and {line_number}")
                    duplicate_vectors += 1
                else:
                    seen_vectors[key] = (role, label, line_number)
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"CSV row {line_number}: {exc}") from exc
            seen_ids.add(sample_id)
            features.append(vector)
            labels.append(label)
            sample_ids.append(sample_id)
            session_ids.append(session)
            counts[(session, label)] += 1
    if (file_sha256(path) != original_hash or file_sha256(meta_path) != meta_hash
            or file_sha256(lineage_path) != lineage_hash):
        raise ValueError("Dataset changed during checking. Close the collector and run the check again.")
    if set(origins) != seen_ids:
        raise ValueError("Lineage sample IDs differ from the merged dataset")
    for session in session_to_split:
        for label in SESSION_LABELS[session]:
            name = CLASS_LABELS[label]
            count = counts[(session, label)]
            if count != expected_per_class:
                raise ValueError(f"{session}/{name}: found {count} rows; expected {expected_per_class}. Report this before changing the dataset.")
    session_array = np.asarray(session_ids)
    split_indices = {
        role: np.flatnonzero(np.isin(session_array, sessions))
        for role, sessions in config["split_sessions"].items()
    }
    for role, indices in split_indices.items():
        if set(np.asarray(labels)[indices].tolist()) != set(range(len(CLASS_LABELS))):
            raise ValueError(f"Every class must appear in the {role} split")
    session_counts = {
        session: {name: counts[(session, label)] for label, name in enumerate(CLASS_LABELS)}
        for session in session_to_split
    }
    return ValidatedDataset(
        path.resolve(), original_hash, meta_hash, metadata,
        np.stack(features), np.asarray(labels, dtype=np.int64), sample_ids,
        session_ids, split_indices, session_counts, duplicate_vectors, expected_per_class,
    )


def make_manifest(data: ValidatedDataset, config: dict, *, project_root: Path = ROOT) -> dict:
    try:
        csv_name = data.path.relative_to(Path(project_root).resolve()).as_posix()
    except ValueError:
        csv_name = str(data.path)
    splits = {}
    for role, indices in data.split_indices.items():
        class_counts = np.bincount(data.labels[indices], minlength=len(CLASS_LABELS))
        splits[role] = {
            "session_ids": config["split_sessions"][role],
            "row_count": len(indices),
            "class_counts": {name: int(class_counts[i]) for i, name in enumerate(CLASS_LABELS)},
            "sample_ids": [data.sample_ids[int(i)] for i in indices],
        }
    return {
        "manifest_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_csv": csv_name,
        "dataset_sha256": data.sha256,
        "metadata_sha256": data.metadata_sha256,
        "feature_version": FEATURE_VERSION,
        "gesture_set_version": GESTURE_SET_VERSION, "gesture_poses": GESTURE_POSES,
        "lineage_sha256": data.metadata["lineage_sha256"],
        "raw_sources": data.metadata["raw_sources"],
        "feature_count": FEATURE_COUNT,
        "class_labels": list(CLASS_LABELS),
        "physical_hand": data.metadata["physical_hand"],
        "camera_index": data.metadata["camera_index"],
        "input_orientation": data.metadata["input_orientation"],
        "total_rows": len(data.sample_ids),
        "unique_sample_ids": len(set(data.sample_ids)),
        "expected_rows_per_session_class": data.expected_per_class,
        "session_counts": data.session_counts,
        "duplicate_feature_rows_within_split": data.duplicate_feature_rows_within_split,
        "exact_feature_duplicates_across_splits": 0,
        "splits": splits,
    }


def save_manifest(path: Path, manifest: dict) -> bool:
    """Create a manifest, keep an identical one, and refuse stale replacements."""
    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(existing, dict):
            raise ValueError(f"Existing {path} is not a valid manifest object; keep it and report this error.")
        left = {key: value for key, value in existing.items() if key != "created_at_utc"}
        right = {key: value for key, value in manifest.items() if key != "created_at_utc"}
        if left != right:
            raise ValueError(f"Existing {path} describes different data/settings. Keep it and report this mismatch before replacing anything.")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return True
