"""Train the separate four-class index-up model; select by validation loss and save CPU weights."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import tempfile
import time

import numpy as np

from dataset import EXPECTED_SPLITS, load_dataset, read_config
from hand_features import CLASS_LABELS, FEATURE_VERSION, GESTURE_SET_VERSION, GESTURE_POSES
from palm_features import MODEL_INPUT_VERSION, to_model_input
from reporting import training_plot
from training_utils import ARCHITECTURE, EarlyStopping, package_versions, project_path, require_manifest, training_settings, write_json


def fit(model, train_x, train_y, validation_x, validation_y, settings):
    """Fit with train/validation tensors only; test tensors are never supplied."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=settings["batch_size"],
                        shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(settings["seed"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    stopping = EarlyStopping(settings["patience"])
    history, best_state = [], None
    for epoch in range(1, settings["max_epochs"] + 1):
        model.train()
        total_loss, correct = 0.0, 0
        for features, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss).item():
                raise ValueError("Training loss is not finite")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(targets)
            correct += int((logits.argmax(dim=1) == targets).sum().item())
        model.eval()
        with torch.inference_mode():
            validation_logits = model(validation_x)
            validation_loss = float(criterion(validation_logits, validation_y).item())
            validation_accuracy = float((validation_logits.argmax(dim=1) == validation_y).float().mean().item())
        row = {"epoch": epoch, "train_loss": total_loss / len(train_y), "train_accuracy": correct / len(train_y),
               "validation_loss": validation_loss, "validation_accuracy": validation_accuracy}
        history.append(row)
        if stopping.update(validation_loss, epoch):
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or stopping.should_stop or epoch == settings["max_epochs"]:
            print(f"Epoch {epoch:03d}: train loss={row['train_loss']:.5f} acc={row['train_accuracy']:.3f} | validation loss={validation_loss:.5f} acc={validation_accuracy:.3f}", flush=True)
        if stopping.should_stop:
            print(f"Early stopping: {settings['patience']} epochs without validation-loss improvement.", flush=True)
            break
    if best_state is None:
        raise ValueError("No valid checkpoint was selected")
    model.load_state_dict(best_state)
    model.eval()
    return best_state, history, stopping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output", default="models/gesture_mlp.pt")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--device-label", choices=("laptop", "desktop", "unspecified"), default="unspecified")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly back up and replace an existing checkpoint.")
    args = parser.parse_args()
    try:
        import torch
        from model import GestureMLP, load_gesture_model
    except ImportError as exc:
        raise RuntimeError("Run with the verified project .venv Python; CPU PyTorch is required.") from exc
    config = read_config(project_path(args.config))
    settings = training_settings(config)
    data = load_dataset(project_path(config["dataset_path"]), config)
    manifest = require_manifest(data, config)
    output, reports = project_path(args.output), project_path(args.reports_dir)
    if output.exists() and not args.overwrite:
        raise ValueError("A checkpoint already exists. Keep it, or deliberately rerun with --overwrite to back it up and replace it.")
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    torch.set_num_threads(settings["num_threads"])
    torch.use_deterministic_algorithms(True)
    model = GestureMLP().to("cpu")
    train_ids, validation_ids = data.split_indices["train"], data.split_indices["validation"]
    train_x, train_y = torch.from_numpy(to_model_input(data.features[train_ids])), torch.from_numpy(data.labels[train_ids])
    validation_x, validation_y = torch.from_numpy(to_model_input(data.features[validation_ids])), torch.from_numpy(data.labels[validation_ids])
    print(f"Model inputs: {MODEL_INPUT_VERSION}; existing raw CSVs are aligned in memory.", flush=True)
    print(f"CPU training on {args.device_label}; {sum(p.numel() for p in model.parameters())} parameters; seed {settings['seed']}", flush=True)
    print(f"Weight updates: {len(train_ids)} training rows. Checkpoint selection: {len(validation_ids)} validation rows.", flush=True)
    print("simple_test01 is reserved; no test predictions or test metrics are computed during training.", flush=True)
    started = time.perf_counter()
    state, history, stopping = fit(model, train_x, train_y, validation_x, validation_y, settings)
    seconds = time.perf_counter() - started
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_input_version": MODEL_INPUT_VERSION,
        "feature_version": FEATURE_VERSION, "class_labels": list(CLASS_LABELS), "architecture": ARCHITECTURE,
        "gesture_set_version": GESTURE_SET_VERSION, "gesture_poses": GESTURE_POSES,
        "training_settings": settings, "seed": settings["seed"], "selected_epoch": stopping.best_epoch,
        "best_validation_loss": stopping.best_loss, "epochs_completed": len(history), "fit_seconds": seconds,
        "device": "cpu", "device_label": args.device_label, "host_platform": platform.platform(),
        "python_version": platform.python_version(), "package_versions": package_versions(),
        "dataset_sha256": data.sha256, "split_sessions": EXPECTED_SPLITS, "split_manifest": manifest,
        "physical_hand": data.metadata["physical_hand"], "input_orientation": data.metadata["input_orientation"],
        "loss": "CrossEntropyLoss on logits", "optimizer": "Adam",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    if output.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = output.with_name(f"{output.stem}.{stamp}.bak{output.suffix}")
        shutil.copy2(output, backup)
        print(f"Previous checkpoint backed up: {backup}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
            temporary = Path(handle.name)
        torch.save({"checkpoint_version": 2, "state_dict": state, "metadata": metadata}, temporary)
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    reloaded, _ = load_gesture_model(output, expected_dataset_sha256=data.sha256)
    with torch.inference_mode():
        original_logits, restored_logits = model(validation_x), reloaded(validation_x)
    if not torch.allclose(original_logits, restored_logits, rtol=0, atol=0):
        raise RuntimeError("Saved/reloaded logits differ; report this before using the checkpoint")
    with (reports / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    write_json(reports / "training_summary.json", metadata)
    training_plot(history, stopping.best_epoch, reports / "training_curves.png")
    print(f"Selected epoch: {stopping.best_epoch}")
    print(f"Best validation loss: {stopping.best_loss:.6f}")
    print(f"Fit time: {seconds:.2f} seconds on {args.device_label} CPU")
    print(f"Checkpoint: {output}")
    print("Checkpoint reload check: PASSED")
    print("TRAINING COMPLETE. Next: evaluate.py --split validation in a fresh process.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining stopped by user; no completion is claimed.", file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"TRAINING FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
