"""Evaluate a saved binary X-ray classifier on the untouched predefined test set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve

from preprocessing import CLASS_NAMES, load_test_records, make_dataset


def metric_payload(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[dict[str, float | int], np.ndarray]:
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    metrics = {
        "test_image_count": int(len(y_true)), "threshold": 0.5,
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
        "accuracy": float((tp + tn) / len(y_true)), "precision_pneumonia": float(precision),
        "recall_sensitivity_pneumonia": float(recall), "specificity_normal": float(specificity),
        "f1_pneumonia": float(f1), "roc_auc": float(auc(fpr, tpr)),
    }
    return metrics, predictions


def save_confusion_matrix(matrix: np.ndarray, output_path: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
             xlabel="Predicted class", ylabel="True class", title=title)
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color="black")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_roc(y_true: np.ndarray, probabilities: np.ndarray, output_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    auc_score = auc(fpr, tpr)
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.plot(fpr, tpr, color="#007C91", label=f"ROC-AUC = {auc_score:.3f}")
    axis.plot([0, 1], [0, 1], "--", color="gray", label="No-skill reference")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC Curve")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def evaluate(model_path: Path, dataset_root: Path, output_dir: Path, image_size: int, batch_size: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(model_path)
    records = load_test_records(dataset_root)
    test_dataset = make_dataset(records, image_size, batch_size, training=False)
    y_true = np.array([int(record["label"]) for record in records])
    probabilities = model.predict(test_dataset, verbose=1).reshape(-1)
    metrics, predictions = metric_payload(y_true, probabilities)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    metrics["model_path"] = str(model_path.resolve())
    metrics["dataset_root"] = str(dataset_root.resolve())
    (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "classification_report.txt").write_text(
        classification_report(y_true, predictions, target_names=CLASS_NAMES, digits=4, zero_division=0), encoding="utf-8"
    )
    with (output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("file", "true_class", "true_label", "pneumonia_probability", "predicted_class", "predicted_label"))
        writer.writeheader()
        for record, true_label, probability, predicted_label in zip(records, y_true, probabilities, predictions):
            writer.writerow({
                "file": record["path"], "true_class": CLASS_NAMES[true_label], "true_label": true_label,
                "pneumonia_probability": f"{probability:.8f}", "predicted_class": CLASS_NAMES[predicted_label],
                "predicted_label": predicted_label,
            })
    save_confusion_matrix(matrix, output_dir / "confusion_matrix.png", "Test-set Confusion Matrix")
    save_roc(y_true, probabilities, output_dir / "roc_curve.png")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    arguments = parser.parse_args()
    evaluate(**vars(arguments))


if __name__ == "__main__":
    main()
