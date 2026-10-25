"""Reload a CPU checkpoint and report validation or final held-out test metrics."""

import argparse
import csv
from datetime import datetime, timezone
import platform
import sys

import numpy as np

from dataset import file_sha256, load_dataset, read_config
from hand_features import CLASS_LABELS
from reporting import confusion_plot, summarize_predictions
from training_utils import package_versions, project_path, require_manifest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--checkpoint", default="models/gesture_mlp.pt")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()
    try:
        import torch
        from model import load_gesture_model
    except ImportError as exc:
        raise RuntimeError("Run with the verified project .venv Python; CPU PyTorch is required.") from exc
    config = read_config(project_path(args.config))
    data = load_dataset(project_path(config["dataset_path"]), config)
    manifest = require_manifest(data, config)
    checkpoint = project_path(args.checkpoint)
    model, training = load_gesture_model(checkpoint, expected_dataset_sha256=data.sha256)
    torch.set_num_threads(training["training_settings"]["num_threads"])
    indices = data.split_indices[args.split]
    truth = data.labels[indices]
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(data.features[indices]))
        loss = float(torch.nn.functional.cross_entropy(logits, torch.from_numpy(truth)).item())
        scores = torch.softmax(logits, dim=1).numpy()
    if not np.isfinite(scores).all() or not np.isfinite(loss):
        raise ValueError("Evaluation produced non-finite outputs")
    predicted = scores.argmax(axis=1)
    summary, text_report = summarize_predictions(truth, predicted)
    summary.update({
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(), "split": args.split,
        "session_ids": config["split_sessions"][args.split], "sample_count": len(indices),
        "session_count": len(config["split_sessions"][args.split]), "class_labels": list(CLASS_LABELS),
        "cross_entropy_loss": loss, "dataset_sha256": data.sha256,
        "checkpoint_sha256": file_sha256(checkpoint), "selected_epoch": training["selected_epoch"],
        "feature_version": training["feature_version"], "physical_hand": data.metadata["physical_hand"],
        "gesture_set_version": training["gesture_set_version"], "gesture_poses": training["gesture_poses"],
        "training_device_label": training["device_label"], "evaluation_device": "cpu",
        "host_platform": platform.platform(), "package_versions": package_versions(),
        "split_row_counts": {role: item["row_count"] for role, item in manifest["splits"].items()},
        "score_note": "Softmax scores are uncalibrated model scores; metrics use argmax without a controller threshold.",
        "scope": "One person's right hand, session-held-out evaluation; no multi-person claim.",
    })
    reports = project_path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    metric_name = "metrics.json" if args.split == "test" else "validation_metrics.json"
    matrix_name = "confusion_matrix.png" if args.split == "test" else "validation_confusion_matrix.png"
    report_name = "classification_report.txt" if args.split == "test" else "validation_classification_report.txt"
    write_json(reports / metric_name, summary)
    (reports / report_name).write_text(text_report, encoding="utf-8")
    columns = ["sample_id", "session_id", "true_id", "true_label", "predicted_id", "predicted_label", "model_score", *[f"score_{name}" for name in CLASS_LABELS]]
    with (reports / f"{args.split}_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for offset, index in enumerate(indices):
            actual, prediction = int(truth[offset]), int(predicted[offset])
            writer.writerow([data.sample_ids[int(index)], data.session_ids[int(index)], actual, CLASS_LABELS[actual],
                             prediction, CLASS_LABELS[prediction], float(scores[offset, prediction]), *scores[offset].tolist()])
    confusion_plot(summary["confusion_matrix"], args.split, reports / matrix_name)
    print(f"Split: {args.split} | sessions: {', '.join(summary['session_ids'])} | rows: {len(indices)}")
    print(f"Accuracy: {summary['accuracy']:.4f} ({summary['accuracy']*100:.2f}%)")
    print(f"Macro F1: {summary['macro_f1']:.4f}")
    print(f"Cross-entropy loss: {loss:.6f}")
    print(text_report)
    print(f"Metrics: {reports / metric_name}")
    print(f"Predictions and confusion matrix saved in: {reports}")
    print("EVALUATION COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEvaluation stopped by user; no completion is claimed.", file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"EVALUATION FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
