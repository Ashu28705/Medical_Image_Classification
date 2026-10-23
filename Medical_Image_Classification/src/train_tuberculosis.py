"""Train a separate ResNet50 tuberculosis classifier from the prepared manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
os.environ.setdefault("KERAS_HOME", str(WORKSPACE_ROOT / ".keras_cache_tuberculosis"))
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from gradcam import make_gradcam_heatmap, overlay_heatmap

CLASS_NAMES = ("NORMAL", "TUBERCULOSIS")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
SPLITS = ("train", "validation", "test")
IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 4
SEED = 42
LEARNING_RATE = 1e-4


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def read_manifest(dataset_root: Path) -> dict[str, list[dict[str, Any]]]:
    report_path = dataset_root / "validation_report.json"
    manifest_path = dataset_root / "split_manifest.csv"
    if not report_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Expected split_manifest.csv and validation_report.json in {dataset_root}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("validation_passed") is not True:
        raise ValueError(f"Prepared dataset validation did not pass: {report_path}")
    if report.get("duplicate_leakage_group_count") != 0:
        raise ValueError("Prepared manifest validation reports duplicate leakage")

    records: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    seen_paths: set[str] = set()
    duplicate_group_splits: defaultdict[str, set[str]] = defaultdict(set)
    with manifest_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"image_path", "class_label", "split", "source", "duplicate_group_sha256"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Manifest lacks required fields: {sorted(required)}")
        for row in reader:
            split = row["split"].strip().casefold()
            class_name = row["class_label"].strip().upper()
            relative_path = Path(row["image_path"])
            if split not in records or class_name not in CLASS_TO_INDEX:
                raise ValueError(f"Unexpected split or class in manifest row: {row}")
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"Manifest path escapes prepared dataset: {row['image_path']}")
            image_path = (dataset_root / relative_path).resolve()
            try:
                image_path.relative_to(dataset_root.resolve())
            except ValueError as error:
                raise ValueError(f"Manifest path escapes prepared dataset: {row['image_path']}") from error
            if not image_path.is_file():
                raise FileNotFoundError(f"Manifest image is missing: {image_path}")
            if image_path.suffix.casefold() not in {".png", ".jpg", ".jpeg"}:
                raise ValueError(f"Unsupported image type in manifest: {image_path}")
            expected_parts = (split, class_name.title())
            if tuple(part.casefold() for part in relative_path.parts[:2]) != tuple(
                part.casefold() for part in expected_parts
            ):
                raise ValueError(f"Manifest path disagrees with its split/class: {row}")
            normalized_path = relative_path.as_posix().casefold()
            if normalized_path in seen_paths:
                raise ValueError(f"Duplicate image path in manifest: {row['image_path']}")
            seen_paths.add(normalized_path)
            group_hash = row.get("duplicate_group_sha256", "").strip()
            if group_hash:
                duplicate_group_splits[group_hash].add(split)
            records[split].append({
                "path": str(image_path),
                "relative_path": relative_path.as_posix(),
                "class_name": class_name,
                "label": CLASS_TO_INDEX[class_name],
                "source": row.get("source", "").strip(),
            })

    leaking_groups = [group for group, split_names in duplicate_group_splits.items() if len(split_names) > 1]
    if leaking_groups:
        raise ValueError(f"Duplicate groups occur across splits: {leaking_groups[:10]}")
    if any(not records[split] for split in SPLITS):
        raise ValueError("Manifest must contain train, validation, and test records")
    for split in SPLITS:
        labels = {record["class_name"] for record in records[split]}
        if labels != set(CLASS_NAMES):
            raise ValueError(f"Split {split} does not contain both expected classes")
    return records


def compute_class_weights(train_records: list[dict[str, Any]]) -> dict[int, float]:
    counts = Counter(record["label"] for record in train_records)
    if set(counts) != {0, 1}:
        raise ValueError(f"Both classes are required in training data; got {counts}")
    total = len(train_records)
    return {label: total / (len(CLASS_NAMES) * counts[label]) for label in CLASS_TO_INDEX.values()}


def make_dataset(
    records: list[dict[str, Any]], batch_size: int, training: bool, seed: int, image_size: int
) -> tf.data.Dataset:
    paths = [record["path"] for record in records]
    labels = np.asarray([record["label"] for record in records], dtype=np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(records), seed=seed, reshuffle_each_iteration=True)

    def decode_and_resize(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [image_size, image_size], antialias=True)
        return tf.cast(image, tf.float32), label

    dataset = dataset.map(decode_and_resize, num_parallel_calls=4, deterministic=True)
    return dataset.batch(batch_size).prefetch(1)


def build_model(image_size: int, seed: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="xray_rgb_0_255")
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.02, seed=seed),
        tf.keras.layers.RandomTranslation(0.02, 0.02, seed=seed + 1),
        tf.keras.layers.RandomContrast(0.08, seed=seed + 2),
    ], name="training_only_augmentation")
    augmented = augmentation(inputs)
    preprocessed = tf.keras.applications.resnet50.preprocess_input(augmented)
    backbone = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=preprocessed,
    )
    backbone.trainable = False
    features = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(backbone.output)
    features = tf.keras.layers.Dropout(0.35, seed=seed, name="classification_dropout")(features)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="tuberculosis_probability")(features)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="tuberculosis_resnet50")


def smoke_test(model_path: Path, sample: dict[str, Any], image_size: int) -> dict[str, Any]:
    model = tf.keras.models.load_model(model_path)
    expected_input = (None, image_size, image_size, 3)
    expected_output = (None, 1)
    if tuple(model.input_shape) != expected_input:
        raise ValueError(f"Unexpected model input shape: {model.input_shape}")
    if tuple(model.output_shape) != expected_output:
        raise ValueError(f"Unexpected model output shape: {model.output_shape}")
    with Image.open(sample["path"]) as image:
        rgb = image.convert("RGB").resize((image_size, image_size))
        batch = np.expand_dims(np.asarray(rgb, dtype=np.float32), axis=0)
    probabilities = model.predict(batch, verbose=0).reshape(-1)
    if probabilities.shape != (1,) or not np.isfinite(probabilities).all():
        raise ValueError(f"Smoke-test prediction is invalid: {probabilities}")
    probability = float(probabilities[0])
    if not 0 <= probability <= 1:
        raise ValueError(f"Smoke-test probability is outside [0, 1]: {probability}")
    return {
        "model_loaded": True,
        "input_shape": list(model.input_shape),
        "output_shape": list(model.output_shape),
        "test_image": sample["relative_path"],
        "test_image_class": sample["class_name"],
        "tuberculosis_probability": probability,
        "probability_finite_and_bounded": True,
    }


def evaluate_test_set(
    model: tf.keras.Model,
    records: list[dict[str, Any]],
    batch_size: int,
    seed: int,
    image_size: int,
    output_dir: Path,
) -> dict[str, Any]:
    test_dataset = make_dataset(records, batch_size, training=False, seed=seed, image_size=image_size)
    probabilities = model.predict(test_dataset, verbose=1).reshape(-1)
    y_true = np.asarray([record["label"] for record in records], dtype=np.int32)
    if len(probabilities) != len(y_true) or not np.isfinite(probabilities).all():
        raise ValueError("Test predictions contain missing or non-finite probabilities")
    y_pred = (probabilities >= 0.5).astype(np.int32)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    per_class = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    text_report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES,
        digits=4, zero_division=0,
    )
    metrics = {
        "test_image_count": int(len(y_true)),
        "threshold": 0.5,
        "positive_class_for_precision_recall_f1": "TUBERCULOSIS",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_tuberculosis": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_tuberculosis": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_tuberculosis": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix_labels": list(CLASS_NAMES),
        "confusion_matrix_true_rows_predicted_columns": matrix.tolist(),
        "per_class_results": per_class,
        "patient_level_separation": "not guaranteed; patient identifiers are unavailable",
        "clinical_validation": "not performed; research use only",
    }
    (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "classification_report.txt").write_text(text_report, encoding="utf-8")
    with (output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=(
            "image_path", "true_class", "true_label", "tuberculosis_probability",
            "predicted_class", "predicted_label",
        ))
        writer.writeheader()
        for record, true_label, probability, predicted_label in zip(records, y_true, probabilities, y_pred):
            writer.writerow({
                "image_path": record["relative_path"],
                "true_class": record["class_name"],
                "true_label": int(true_label),
                "tuberculosis_probability": f"{float(probability):.8f}",
                "predicted_class": CLASS_NAMES[int(predicted_label)],
                "predicted_label": int(predicted_label),
            })
    return metrics


def create_gradcam(
    model: tf.keras.Model, sample: dict[str, Any], output_dir: Path, image_size: int
) -> dict[str, Any]:
    with Image.open(sample["path"]) as image:
        original = np.asarray(image.convert("RGB").resize((image_size, image_size)))
    batch = np.expand_dims(original.astype(np.float32), axis=0)
    heatmap, probability, layer_name = make_gradcam_heatmap(batch, model)
    if heatmap.ndim != 2 or not np.isfinite(heatmap).all() or float(np.max(heatmap)) <= 0:
        raise ValueError("Grad-CAM returned an empty, non-finite, or invalid heatmap")
    colored, overlay = overlay_heatmap(original, heatmap)
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(original).save(output_dir / "test_image.png")
    Image.fromarray(colored).save(output_dir / "test_gradcam.png")
    Image.fromarray(overlay).save(output_dir / "test_overlay.png")
    return {
        "generated": True,
        "test_image": sample["relative_path"],
        "target_class": "TUBERCULOSIS",
        "probability": float(probability),
        "layer": layer_name,
        "heatmap_shape": list(heatmap.shape),
        "heatmap_finite": bool(np.isfinite(heatmap).all()),
    }


def train(args: argparse.Namespace) -> None:
    set_reproducible_seed(args.seed)
    cpu_count = os.cpu_count() or 1
    try:
        tf.config.threading.set_intra_op_parallelism_threads(min(cpu_count, 8))
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass
    gpu_devices = tf.config.list_physical_devices("GPU")

    dataset_root = args.dataset_root.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Prepared dataset directory not found: {dataset_root}")
    if results_dir.exists():
        raise FileExistsError(f"TB results directory already exists; refusing to overwrite: {results_dir}")
    if model_path.exists():
        raise FileExistsError(f"TB model artifact already exists; refusing to overwrite: {model_path}")

    splits = read_manifest(dataset_root)
    train_records = splits["train"]
    validation_records = splits["validation"]
    test_records = splits["test"]
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 32:
        raise ValueError("epochs and batch size must be positive; image size must be at least 32")
    class_weights = compute_class_weights(train_records)
    train_counts = Counter(record["class_name"] for record in train_records)
    source_counts = {
        split: dict(Counter(record["source"] or "Unknown" for record in records))
        for split, records in splits.items()
    }

    results_dir.mkdir(parents=True)
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "best_model.keras"
    run_config = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": "ResNet50 transfer learning; frozen ImageNet backbone",
        "dataset_root": str(dataset_root),
        "manifest_path": str(dataset_root / "split_manifest.csv"),
        "split_source": "prepared split_manifest.csv; no new split generated",
        "split_counts": {
            split: dict(Counter(record["class_name"] for record in records))
            for split, records in splits.items()
        },
        "source_counts_by_split": source_counts,
        "classes": {"NORMAL": 0, "TUBERCULOSIS": 1},
        "positive_class": "TUBERCULOSIS",
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "seed": args.seed,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "class_weights_from_training_split_only": {
            CLASS_NAMES[index]: weight for index, weight in class_weights.items()
        },
        "augmentation": {
            "layers": ["RandomRotation(0.02)", "RandomTranslation(0.02, 0.02)", "RandomContrast(0.08)"],
            "active_during_training_only": True,
            "horizontal_or_vertical_flip": False,
        },
        "preprocessing": "RGB float pixels in [0,255], resize to image_size, training augmentation, then tf.keras.applications.resnet50.preprocess_input",
        "tensorflow_version": tf.__version__,
        "python_version": platform.python_version(),
        "cpu_count": cpu_count,
        "gpus": [device.name for device in gpu_devices],
        "hardware_note": "CPU-only training" if not gpu_devices else "GPU available",
        "patient_level_separation": "not guaranteed; patient identifiers are unavailable",
    }
    (results_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print("TB RESNET50 TRAINING")
    print(f"  TensorFlow {tf.__version__} | CPUs {cpu_count} | GPUs {gpu_devices or 'none'}")
    print(f"  Train {len(train_records)} {dict(train_counts)} | Validation {len(validation_records)} | Test reserved {len(test_records)}")
    print(f"  Class weights from training only: {run_config['class_weights_from_training_split_only']}")
    print(f"  Image size {args.image_size}, batch {args.batch_size}, max epochs {args.epochs}, seed {args.seed}")

    train_dataset = make_dataset(
        train_records, args.batch_size, training=True, seed=args.seed, image_size=args.image_size
    )
    validation_dataset = make_dataset(
        validation_records, args.batch_size, training=False, seed=args.seed, image_size=args.image_size
    )
    model = build_model(args.image_size, args.seed)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_loss", mode="min", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=2, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=1, min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(results_dir / "training_history.csv"),
    ]
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=2,
    )
    if not checkpoint_path.is_file():
        raise RuntimeError("Training finished without a validation-selected model checkpoint")
    run_config["epochs_completed"] = len(history.history.get("loss", []))
    run_config["best_validation_loss"] = float(min(history.history["val_loss"]))
    (results_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    best_model = tf.keras.models.load_model(checkpoint_path)
    test_sample = next(record for record in test_records if record["class_name"] == "TUBERCULOSIS")
    smoke_result = smoke_test(checkpoint_path, test_sample, args.image_size)
    gradcam_result = create_gradcam(best_model, test_sample, results_dir / "gradcam", args.image_size)
    metrics = evaluate_test_set(
        best_model, test_records, args.batch_size, args.seed, args.image_size, results_dir
    )
    smoke_result["gradcam"] = gradcam_result
    (results_dir / "smoke_test.json").write_text(json.dumps(smoke_result, indent=2), encoding="utf-8")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_path, model_path)
    try:
        final_smoke = smoke_test(model_path, test_sample, args.image_size)
    except Exception:
        model_path.unlink(missing_ok=True)
        raise
    label_mapping = {
        "class_to_index": {"NORMAL": 0, "TUBERCULOSIS": 1},
        "positive_class": "TUBERCULOSIS",
        "patient_ids_available": False,
    }
    preprocessing_config = {
        "input_color_space": "RGB",
        "input_pixel_range_before_preprocess": [0, 255],
        "resize": [args.image_size, args.image_size],
        "preprocess_function": "tf.keras.applications.resnet50.preprocess_input",
        "augmentation": "training only: RandomRotation(0.02), RandomTranslation(0.02, 0.02), RandomContrast(0.08)",
        "prediction_threshold": 0.5,
    }
    model_path.parent.joinpath("label_mapping.json").write_text(
        json.dumps(label_mapping, indent=2), encoding="utf-8"
    )
    model_path.parent.joinpath("preprocessing_config.json").write_text(
        json.dumps(preprocessing_config, indent=2), encoding="utf-8"
    )
    model_path.parent.joinpath("run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    (results_dir / "final_smoke_test.json").write_text(
        json.dumps(final_smoke, indent=2), encoding="utf-8"
    )
    (results_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({
        "saved_model": str(model_path),
        "results_directory": str(results_dir),
        "epochs_completed": run_config["epochs_completed"],
        "test_metrics": metrics,
        "smoke_test": final_smoke,
        "gradcam": gradcam_result,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path,
        default=WORKSPACE_ROOT / "TB_Chest_Radiography_Database_prepared",
    )
    parser.add_argument(
        "--results-dir", type=Path,
        default=PROJECT_ROOT / "results" / "tuberculosis",
    )
    parser.add_argument(
        "--model-path", type=Path,
        default=PROJECT_ROOT / "models" / "tuberculosis" / "best_model.keras",
    )
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    train(parser.parse_args())


if __name__ == "__main__":
    main()