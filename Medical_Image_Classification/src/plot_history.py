"""Render loss and metric curves from the CSV written by Keras CSVLogger."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_history(history_path: Path, output_dir: Path) -> None:
    history = pd.read_csv(history_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric, title, filename in (("accuracy", "Training and Validation Accuracy", "accuracy_curve.png"),
                                    ("loss", "Training and Validation Loss", "loss_curve.png")):
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.plot(history["epoch"] + 1, history[metric], marker="o", label=f"Training {metric}")
        validation_metric = f"val_{metric}"
        if validation_metric in history:
            axis.plot(history["epoch"] + 1, history[validation_metric], marker="o", label=f"Validation {metric}")
        axis.set(xlabel="Epoch", ylabel=metric.capitalize(), title=title)
        axis.legend()
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=180)
        plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    plot_history(arguments.history, arguments.output_dir)
