"""Reproducible, duplicate-aware tf.data pipelines for the X-ray project."""

from __future__ import annotations

import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf

CLASS_NAMES = ("NORMAL", "PNEUMONIA")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}


def set_global_seed(seed: int = 42) -> None:
    """Set practical seeds for repeatable file splitting and model initialization."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _direct_image_paths(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def build_training_records(dataset_root: Path) -> list[dict[str, str | int]]:
    """Create records from direct train folders only, excluding any nested dataset copy."""
    records: list[dict[str, str | int]] = []
    for class_name, label in CLASS_TO_INDEX.items():
        for path in _direct_image_paths(dataset_root / "train" / class_name):
            records.append({"path": str(path), "label": label, "hash": _sha256(path)})
    return records


def duplicate_aware_train_validation_split(
    records: list[dict[str, str | int]], validation_fraction: float = 0.15, seed: int = 42
) -> tuple[list[dict[str, str | int]], list[dict[str, str | int]]]:
    """Stratify by class while keeping byte-identical images together in one subset."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    by_class: dict[int, list[dict[str, str | int]]] = defaultdict(list)
    for record in records:
        by_class[int(record["label"])].append(record)

    train_records: list[dict[str, str | int]] = []
    validation_records: list[dict[str, str | int]] = []
    for label, class_records in by_class.items():
        groups: dict[str, list[dict[str, str | int]]] = defaultdict(list)
        for record in class_records:
            groups[str(record["hash"])].append(record)
        group_list = list(groups.values())
        random.Random(seed + label).shuffle(group_list)
        target_validation_count = round(len(class_records) * validation_fraction)
        validation_count = 0
        for group in group_list:
            if validation_count < target_validation_count:
                validation_records.extend(group)
                validation_count += len(group)
            else:
                train_records.extend(group)
    return train_records, validation_records


def write_split_manifest(records: Iterable[dict[str, str | int]], split_name: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exists = output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("subset", "path", "label", "class_name", "sha256"))
        if not exists:
            writer.writeheader()
        for record in records:
            label = int(record["label"])
            writer.writerow({
                "subset": split_name,
                "path": record["path"],
                "label": label,
                "class_name": CLASS_NAMES[label],
                "sha256": record["hash"],
            })


def _decode_resize(path: tf.Tensor, label: tf.Tensor, image_size: int) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [image_size, image_size], antialias=True)
    return tf.cast(image, tf.float32), tf.cast(label, tf.float32)


def make_dataset(
    records: list[dict[str, str | int]], image_size: int, batch_size: int, training: bool, seed: int = 42
) -> tf.data.Dataset:
    paths = [str(record["path"]) for record in records]
    labels = [int(record["label"]) for record in records]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(records), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(
        lambda path, label: _decode_resize(path, label, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def load_test_records(dataset_root: Path) -> list[dict[str, str | int]]:
    records: list[dict[str, str | int]] = []
    for class_name, label in CLASS_TO_INDEX.items():
        for path in _direct_image_paths(dataset_root / "test" / class_name):
            records.append({"path": str(path), "label": label, "hash": _sha256(path)})
    return records


def compute_class_weights(records: list[dict[str, str | int]]) -> dict[int, float]:
    counts = np.bincount([int(record["label"]) for record in records], minlength=2)
    total = counts.sum()
    return {index: float(total / (len(CLASS_NAMES) * count)) for index, count in enumerate(counts)}
