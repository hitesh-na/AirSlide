"""Metric summaries and exportable plots; no model fitting occurs here."""

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from hand_features import CLASS_LABELS


def summarize_predictions(truth, predicted):
    truth = np.asarray(truth, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    if truth.ndim != 1 or truth.shape != predicted.shape or len(truth) == 0:
        raise ValueError("Predictions and targets must be matching nonempty one-dimensional arrays")
    if not np.isin(truth, range(5)).all() or not np.isin(predicted, range(5)).all():
        raise ValueError("Gesture IDs must be 0-4")
    labels = list(range(5))
    options = dict(labels=labels, target_names=list(CLASS_LABELS), zero_division=0)
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)),
        "classification_report": classification_report(truth, predicted, output_dict=True, **options),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
    }, classification_report(truth, predicted, digits=4, **options)


def training_plot(history, selected_epoch, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for axis, metric, title in zip(axes, ("loss", "accuracy"), ("Cross-entropy loss", "Accuracy")):
        for role, color in (("train", "#2563eb"), ("validation", "#0f766e")):
            axis.plot(epochs, [row[f"{role}_{metric}"] for row in history], label=role, color=color)
        axis.axvline(selected_epoch, linestyle="--", color="#64748b", label="selected epoch")
        axis.set(xlabel="Epoch", ylabel=title, title=title)
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def confusion_plot(matrix, split, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matrix = np.asarray(matrix)
    fig, axis = plt.subplots(figsize=(7, 6), layout="constrained")
    image = axis.imshow(matrix, cmap="Blues", vmin=0)
    axis.set(xticks=range(5), yticks=range(5), xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
             xlabel="Predicted gesture", ylabel="Actual gesture", title=f"{split.capitalize()} confusion matrix")
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right")
    threshold = matrix.max() / 2
    for row in range(5):
        for col in range(5):
            axis.text(col, row, str(matrix[row, col]), ha="center", va="center", color="white" if matrix[row, col] > threshold else "black")
    fig.colorbar(image, ax=axis, label="Samples")
    fig.savefig(path, dpi=160)
    plt.close(fig)
