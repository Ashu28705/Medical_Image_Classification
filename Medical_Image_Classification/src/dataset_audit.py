"""Audit the canonical Chest X-Ray Pneumonia dataset without modifying it."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError

SPLITS = ("train", "val", "test")
CLASSES = ("NORMAL", "PNEUMONIA")
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}


def image_files(folder: Path) -> list[Path]:
    """Return only direct image files; do not accidentally enter nested datasets."""
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_class_chart(counts: dict[str, dict[str, int]], output: Path) -> None:
    labels = [f"{split}\n{label}" for split in SPLITS for label in CLASSES]
    values = [counts[split][label] for split in SPLITS for label in CLASSES]
    colors = ["#5BA6A6" if "NORMAL" in label else "#F08A7C" for label in labels]
    figure, axis = plt.subplots(figsize=(10, 5))
    bars = axis.bar(labels, values, color=colors)
    axis.set_title("Chest X-Ray Dataset Class Distribution")
    axis.set_ylabel("Number of images")
    axis.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_samples(samples: dict[str, list[Path]], output: Path) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(12, 6))
    for row, label in enumerate(CLASSES):
        for column, path in enumerate(samples[label][:4]):
            with Image.open(path) as image:
                axes[row, column].imshow(image.convert("L"), cmap="gray")
            axes[row, column].set_title(label)
            axes[row, column].axis("off")
    figure.suptitle("Training Set X-Ray Samples", fontsize=14)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def audit(dataset_root: Path, output_dir: Path) -> dict:
    if not all((dataset_root / split / label).is_dir() for split in SPLITS for label in CLASSES):
        raise FileNotFoundError(
            "Expected train/, val/, and test/ with NORMAL/ and PNEUMONIA/ folders under "
            f"{dataset_root}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {split: {} for split in SPLITS}
    dimensions: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    extensions: Counter[str] = Counter()
    corrupt_files: list[str] = []
    unusual_files: list[str] = []
    hashes: defaultdict[str, list[str]] = defaultdict(list)
    records: list[dict[str, str | int]] = []
    all_paths: list[Path] = []

    for split in SPLITS:
        for label in CLASSES:
            paths = image_files(dataset_root / split / label)
            counts[split][label] = len(paths)
            all_paths.extend(paths)
            for path in paths:
                relative = path.relative_to(dataset_root).as_posix()
                extensions[path.suffix.lower()] += 1
                try:
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        width, height = image.size
                        mode = image.mode
                    if width < 32 or height < 32:
                        unusual_files.append(f"{relative}: extremely small ({width}x{height})")
                    dimensions[f"{width}x{height}"] += 1
                    modes[mode] += 1
                    digest = sha256(path)
                    hashes[digest].append(relative)
                    records.append({
                        "split": split, "class": label, "file": relative,
                        "width": width, "height": height, "mode": mode,
                        "sha256": digest,
                    })
                except (UnidentifiedImageError, OSError, ValueError) as error:
                    corrupt_files.append(f"{relative}: {error}")

    duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    cross_split_duplicates = [paths for paths in duplicates if len({path.split("/")[0] for path in paths}) > 1]
    total = sum(sum(class_counts.values()) for class_counts in counts.values())
    train_total = sum(counts["train"].values())
    train_pneumonia_ratio = counts["train"]["PNEUMONIA"] / train_total

    with (output_dir / "image_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("split", "class", "file", "width", "height", "mode", "sha256"))
        writer.writeheader()
        writer.writerows(records)

    duplicate_payload = {
        "exact_duplicate_groups": duplicates,
        "cross_split_exact_duplicate_groups": cross_split_duplicates,
    }
    (output_dir / "duplicate_report.json").write_text(json.dumps(duplicate_payload, indent=2), encoding="utf-8")

    report = {
        "dataset_root": str(dataset_root.resolve()),
        "scope": "Only direct files in train/val/test class folders were audited. The nested chest_xray copy was excluded.",
        "total_images": total,
        "counts": counts,
        "training_pneumonia_ratio": round(train_pneumonia_ratio, 6),
        "file_extensions": dict(extensions),
        "image_modes": dict(modes),
        "most_common_dimensions": dimensions.most_common(20),
        "corrupt_file_count": len(corrupt_files),
        "corrupt_files": corrupt_files,
        "unusual_file_count": len(unusual_files),
        "unusual_files": unusual_files,
        "exact_duplicate_group_count": len(duplicates),
        "cross_split_exact_duplicate_group_count": len(cross_split_duplicates),
    }
    (output_dir / "dataset_quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    sample_paths = {label: image_files(dataset_root / "train" / label) for label in CLASSES}
    save_class_chart(counts, output_dir / "class_distribution.png")
    save_samples(sample_paths, output_dir / "sample_xrays.png")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.dataset_root, args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
