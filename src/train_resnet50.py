"""End-to-end ResNet50 training pipeline:
1. Train ResNet50 frozen-base head on the train/val split.
2. Fine-tune the last 30 layers at a small learning rate.
3. Evaluate both checkpoints on the test set.
4. Save the best checkpoint to models/resnet50/final_model.keras.
5. Generate Grad-CAM gallery + smoke test.

Designed to be run from the project root as a script via the project's
existing .venv. Reuses preprocessing/models/evaluate/gradcam modules so
preprocessing, class mapping, and Grad-CAM are all consistent with the
Streamlit app.
"""
from __future__ import annotations

# Ensure KERAS_HOME points to the project-local cache so pretrained
# weights are reused (they are already downloaded).
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("KERAS_HOME", str(PROJECT_ROOT / ".keras_cache"))

import matplotlib
matplotlib.use("Agg")  # headless plots

import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf

# Make sibling src/ modules importable when run as a script.
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models import build_resnet50, unfreeze_resnet_top
from preprocessing import (
    CLASS_NAMES,
    build_training_records,
    compute_class_weights,
    duplicate_aware_train_validation_split,
    load_test_records,
    make_dataset,
    set_global_seed,
    write_split_manifest,
)
from evaluate import evaluate as run_evaluation
from gradcam import load_xray, make_gradcam_heatmap, overlay_heatmap
from plot_history import plot_history

DATASET_ROOT = PROJECT_ROOT.parent / "chest_xray"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
FINAL_MODEL_DIR = MODELS_DIR / "resnet50"
FINAL_MODEL_PATH = FINAL_MODEL_DIR / "final_model.keras"

# Configuration (chosen for CPU-only training within reasonable time)
IMAGE_SIZE = 224
BATCH_SIZE = 16
SEED = 42
FROZEN_EPOCHS = 4
FINETUNE_EPOCHS = 2
FINE_TUNE_TOP_LAYERS = 30
FROZEN_LR = 1e-4
FINETUNE_LR = 1e-5

# Use all available CPU threads for tf.data + intra/inter-op parallelism.
_NUM_CORES = os.cpu_count() or 4
tf.config.threading.set_intra_op_parallelism_threads(_NUM_CORES)
tf.config.threading.set_inter_op_parallelism_threads(_NUM_CORES)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_callbacks(output_dir: Path) -> list:
    return [
        tf.keras.callbacks.ModelCheckpoint(
            output_dir / "best_model.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(output_dir / "training_history.csv"),
    ]


def save_run_config(output_dir: Path, model_name: str, lr: float, epochs: int,
                    class_weights: dict, extra: dict | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "timestamp_utc": now_iso(),
        "model": model_name,
        "dataset_root": str(DATASET_ROOT.resolve()),
        "classes": {"NORMAL": 0, "PNEUMONIA": 1},
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_requested": epochs,
        "validation_fraction": 0.15,
        "seed": SEED,
        "optimizer": "Adam",
        "learning_rate": lr,
        "class_weights": class_weights,
        "tensorflow_version": tf.__version__,
        "python_version": platform.python_version(),
        "gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
    }
    if extra:
        config.update(extra)
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def train_one(model: tf.keras.Model, train_ds, val_ds, class_weights: dict,
              epochs: int, output_dir: Path) -> tf.keras.callbacks.History:
    output_dir.mkdir(parents=True, exist_ok=True)
    callbacks = get_callbacks(output_dir)
    print(f"  Training {model.name} for up to {epochs} epoch(s)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=2,
    )
    model.save(output_dir / "last_model.keras")
    (output_dir / "history.json").write_text(
        json.dumps({k: [float(v) for v in vals] for k, vals in history.history.items()}, indent=2),
        encoding="utf-8",
    )
    return history


def gradcam_gallery(model: tf.keras.Model, output_dir: Path, n_per_class: int = 4) -> None:
    import matplotlib.pyplot as plt
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    test_root = DATASET_ROOT / "test"
    samples = []
    for cls in CLASS_NAMES:
        cls_dir = test_root / cls
        if not cls_dir.exists():
            continue
        images = sorted(cls_dir.iterdir())[:n_per_class]
        for img_path in images:
            samples.append((cls, img_path))

    if not samples:
        print("  No test samples found for Grad-CAM gallery.")
        return

    fig, axes = plt.subplots(len(samples), 3, figsize=(12, 4 * len(samples)))
    if len(samples) == 1:
        axes = [axes]

    for idx, (true_class, img_path) in enumerate(samples):
        original, batch = load_xray(str(img_path), IMAGE_SIZE)
        heatmap, prob, layer_name = make_gradcam_heatmap(batch, model)
        colored, overlay_img = overlay_heatmap(original, heatmap)
        pred_class = "PNEUMONIA" if prob >= 0.5 else "NORMAL"
        confidence = prob if pred_class == "PNEUMONIA" else 1.0 - prob

        axes[idx][0].imshow(original)
        axes[idx][0].set_title(f"True: {true_class}", fontsize=10)
        axes[idx][0].axis("off")

        axes[idx][1].imshow(colored)
        axes[idx][1].set_title(f"Grad-CAM ({layer_name})", fontsize=10)
        axes[idx][1].axis("off")

        axes[idx][2].imshow(overlay_img)
        axes[idx][2].set_title(f"Pred: {pred_class} ({confidence:.1%})", fontsize=10)
        axes[idx][2].axis("off")

        Image.fromarray(overlay_img).save(output_dir / f"{true_class}_{img_path.stem}_overlay.png")

    fig.suptitle("Grad-CAM Explainability Gallery", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "gradcam_gallery.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grad-CAM gallery saved to {output_dir}")


def write_comparison(results_dirs: dict, output_dir: Path) -> str | None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_keys = [
        "accuracy",
        "precision_pneumonia",
        "recall_sensitivity_pneumonia",
        "specificity_normal",
        "f1_pneumonia",
        "roc_auc",
    ]
    display_names = [
        "Accuracy", "Precision", "Recall/Sens.",
        "Specificity", "F1", "ROC-AUC",
    ]

    all_metrics = {}
    for name, rdir in results_dirs.items():
        metrics_file = rdir / "evaluation" / "test_metrics.json"
        if metrics_file.exists():
            data = json.loads(metrics_file.read_text(encoding="utf-8"))
            all_metrics[name] = {k: data.get(k, 0.0) for k in metrics_keys}

    if not all_metrics:
        return None

    (output_dir / "comparison_metrics.json").write_text(
        json.dumps(all_metrics, indent=2), encoding="utf-8"
    )

    model_names = list(all_metrics.keys())
    x = np.arange(len(display_names))
    width = 0.8 / max(1, len(model_names))

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    for i, (name, metrics) in enumerate(all_metrics.items()):
        values = [metrics[k] for k in metrics_keys]
        bars = ax.bar(x + i * width, values, width, label=name, color=colors[i % len(colors)], alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("ResNet50 - Test Set Performance (frozen vs fine-tuned)")
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(display_names, rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=180)
    plt.close(fig)

    best_name = max(all_metrics, key=lambda n: all_metrics[n]["f1_pneumonia"])
    (output_dir / "best_model.txt").write_text(
        f"Best model by F1: {best_name}\n" + json.dumps(all_metrics[best_name], indent=2),
        encoding="utf-8",
    )
    return best_name


def main() -> None:
    set_global_seed(SEED)
    print("=" * 72)
    print("RESNET50 TRAINING PIPELINE - chest_xray")
    print(f"  TensorFlow {tf.__version__}  |  Python {platform.python_version()}")
    print(f"  GPUs: {tf.config.list_physical_devices('GPU') or 'None (CPU only)'}")
    print(f"  Dataset: {DATASET_ROOT}")
    print(f"  Image size: {IMAGE_SIZE}  |  Batch size: {BATCH_SIZE}  |  Seed: {SEED}")
    print("=" * 72)

    # 1. Data preparation
    print("\n[1/5] Building data splits...")
    records = build_training_records(DATASET_ROOT)
    train_records, val_records = duplicate_aware_train_validation_split(records, 0.15, SEED)
    class_weights = compute_class_weights(train_records)
    print(f"  Train: {len(train_records)} | Val: {len(val_records)}")
    print(f"  Class distribution (train): NORMAL={sum(1 for r in train_records if r['label']==0)}, "
          f"PNEUMONIA={sum(1 for r in train_records if r['label']==1)}")
    print(f"  Class weights: {class_weights}")

    train_ds = make_dataset(train_records, IMAGE_SIZE, BATCH_SIZE, training=True, seed=SEED)
    val_ds = make_dataset(val_records, IMAGE_SIZE, BATCH_SIZE, training=False, seed=SEED)

    test_records = load_test_records(DATASET_ROOT)
    print(f"  Test: {len(test_records)} "
          f"(NORMAL={sum(1 for r in test_records if r['label']==0)}, "
          f"PNEUMONIA={sum(1 for r in test_records if r['label']==1)})")

    # 2. Train ResNet50 frozen-base head
    frozen_dir = RESULTS_DIR / "resnet50" / "frozen"
    print(f"\n[2/5] Training ResNet50 frozen-base head (<= {FROZEN_EPOCHS} epochs, lr={FROZEN_LR})...")
    frozen_model = build_resnet50(IMAGE_SIZE)
    frozen_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FROZEN_LR),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                 tf.keras.metrics.AUC(name="auc")],
    )
    print(f"  Total params: {frozen_model.count_params():,}")
    trainable = sum(int(np.prod(v.shape)) for v in frozen_model.trainable_variables)
    print(f"  Trainable params: {trainable:,}")

    save_run_config(frozen_dir, "resnet50_frozen", FROZEN_LR, FROZEN_EPOCHS, class_weights,
                    {"base_trainable": False, "pretrained_weights": "imagenet"})

    manifest = frozen_dir / "training_validation_manifest.csv"
    if manifest.exists():
        manifest.unlink()
    write_split_manifest(train_records, "train", manifest)
    write_split_manifest(val_records, "validation", manifest)

    t0 = time.time()
    train_one(frozen_model, train_ds, val_ds, class_weights, FROZEN_EPOCHS, frozen_dir)
    print(f"  Frozen training elapsed: {(time.time() - t0) / 60:.1f} min")

    print("\n[3/5] Evaluating frozen ResNet50 on test set...")
    frozen_eval_dir = frozen_dir / "evaluation"
    run_evaluation(frozen_dir / "best_model.keras", DATASET_ROOT, frozen_eval_dir, IMAGE_SIZE, BATCH_SIZE)
    plot_history(frozen_dir / "training_history.csv", frozen_eval_dir)

    # 3. Fine-tune top layers
    ft_dir = RESULTS_DIR / "resnet50" / "finetuned"
    print(f"\n[4/5] Fine-tuning ResNet50 top {FINE_TUNE_TOP_LAYERS} layers (<= {FINETUNE_EPOCHS} epochs, lr={FINETUNE_LR})...")
    ft_model = tf.keras.models.load_model(frozen_dir / "best_model.keras")
    unfreeze_resnet_top(ft_model, trainable_layers=FINE_TUNE_TOP_LAYERS)
    ft_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINETUNE_LR),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                 tf.keras.metrics.AUC(name="auc")],
    )
    trainable = sum(int(np.prod(v.shape)) for v in ft_model.trainable_variables)
    print(f"  Trainable params after unfreeze: {trainable:,}")
    save_run_config(ft_dir, "resnet50_finetuned", FINETUNE_LR, FINETUNE_EPOCHS, class_weights,
                    {"base_trainable_layers": FINE_TUNE_TOP_LAYERS,
                     "fine_tuned_from": str(frozen_dir / "best_model.keras"),
                     "batchnorm_frozen": True})
    t0 = time.time()
    train_one(ft_model, train_ds, val_ds, class_weights, FINETUNE_EPOCHS, ft_dir)
    print(f"  Fine-tune elapsed: {(time.time() - t0) / 60:.1f} min")

    print("  Evaluating fine-tuned ResNet50 on test set...")
    ft_eval_dir = ft_dir / "evaluation"
    run_evaluation(ft_dir / "best_model.keras", DATASET_ROOT, ft_eval_dir, IMAGE_SIZE, BATCH_SIZE)
    plot_history(ft_dir / "training_history.csv", ft_eval_dir)

    # 4. Compare and pick best
    print("\n[5/5] Comparing checkpoints and writing final model...")
    comparison_dir = RESULTS_DIR / "comparison"
    best_name = write_comparison(
        {"ResNet50 (Frozen)": frozen_dir, "ResNet50 (Fine-tuned)": ft_dir},
        comparison_dir,
    )
    best_map = {
        "ResNet50 (Frozen)": frozen_dir / "best_model.keras",
        "ResNet50 (Fine-tuned)": ft_dir / "best_model.keras",
    }
    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if best_name and best_name in best_map:
        chosen = best_map[best_name]
        print(f"  Best by F1: {best_name} -> {chosen}")
    else:
        chosen = ft_dir / "best_model.keras"
        print(f"  Falling back to fine-tuned checkpoint -> {chosen}")

    shutil.copy2(chosen, FINAL_MODEL_PATH)
    print(f"  Copied to {FINAL_MODEL_PATH}")

    # Verify the saved model reloads and produces a prediction.
    print("  Reloading saved model for verification...")
    reloaded = tf.keras.models.load_model(FINAL_MODEL_PATH)
    sample_normal = next(iter((DATASET_ROOT / "test" / "NORMAL").iterdir()))
    original, batch = load_xray(str(sample_normal), IMAGE_SIZE)
    prob = float(reloaded.predict(batch, verbose=0)[0, 0])
    print(f"  Reload OK. Sample NORMAL test image -> P(PNEUMONIA) = {prob:.4f}")

    # Grad-CAM gallery on best model.
    gradcam_dir = RESULTS_DIR / "gradcam"
    gradcam_gallery(reloaded, gradcam_dir, n_per_class=4)

    # Print the test metrics from the chosen best run for a quick recap.
    metrics_file = (ft_eval_dir if chosen == best_map.get("ResNet50 (Fine-tuned)") else frozen_eval_dir) / "test_metrics.json"
    if metrics_file.exists():
        print("\nFinal test metrics for the chosen model:")
        print(metrics_file.read_text(encoding="utf-8"))

    print("\n" + "=" * 72)
    print("RESNET50 PIPELINE COMPLETE")
    print(f"  Final model: {FINAL_MODEL_PATH}")
    print(f"  Results: {RESULTS_DIR / 'resnet50'}")
    print(f"  Comparison: {comparison_dir}")
    print(f"  Grad-CAM:   {gradcam_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
