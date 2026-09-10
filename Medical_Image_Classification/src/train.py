"""Train either the reproducible CNN baseline or the frozen ResNet50 head."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import tensorflow as tf

from models import build_baseline_cnn, build_resnet50
from preprocessing import (
    build_training_records,
    compute_class_weights,
    duplicate_aware_train_validation_split,
    make_dataset,
    set_global_seed,
    write_split_manifest,
)


def train(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = build_training_records(dataset_root)
    train_records, validation_records = duplicate_aware_train_validation_split(
        records, validation_fraction=args.validation_fraction, seed=args.seed
    )
    split_manifest = output_dir / "training_validation_manifest.csv"
    if split_manifest.exists():
        split_manifest.unlink()
    write_split_manifest(train_records, "train", split_manifest)
    write_split_manifest(validation_records, "validation", split_manifest)
    train_dataset = make_dataset(train_records, args.image_size, args.batch_size, training=True, seed=args.seed)
    validation_dataset = make_dataset(validation_records, args.image_size, args.batch_size, training=False, seed=args.seed)
    class_weights = compute_class_weights(train_records)

    model = build_baseline_cnn(args.image_size) if args.model == "baseline" else build_resnet50(args.image_size)
    learning_rate = args.learning_rate or (1e-3 if args.model == "baseline" else 1e-4)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc")],
    )
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(output_dir / "best_model.keras", monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-7),
        tf.keras.callbacks.CSVLogger(output_dir / "training_history.csv"),
    ]
    run_config = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "dataset_root": str(dataset_root.resolve()),
        "classes": {"NORMAL": 0, "PNEUMONIA": 1},
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "class_weights": class_weights,
        "augmentation": "RandomRotation(0.03), RandomTranslation(0.03, 0.03), RandomContrast(0.08); train only",
        "tensorflow_version": tf.__version__,
        "python_version": platform.python_version(),
        "gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=2,
    )
    model.save(output_dir / "last_model.keras")
    (output_dir / "history.json").write_text(json.dumps(history.history, indent=2), encoding="utf-8")
    print(f"Saved training outputs to {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("baseline", "resnet50"), required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
