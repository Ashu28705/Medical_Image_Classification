"""
Master script: Train baseline CNN, train frozen ResNet50, fine-tune ResNet50,
evaluate all three, compare, generate Grad-CAM, and copy best model.
Run from the src/ directory.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import json
import shutil
import sys
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

# ── Project paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT.parent / "chest_xray"
RESULTS_DIR  = PROJECT_ROOT / "results"
MODELS_DIR   = PROJECT_ROOT / "models"

# ── Configuration ────────────────────────────────────────────────────────────
IMAGE_SIZE   = 224
BATCH_SIZE   = 16
SEED         = 42
CNN_EPOCHS   = 15
RN_FROZEN_EPOCHS  = 10
RN_FINETUNE_EPOCHS = 10

from models import build_baseline_cnn, build_resnet50, unfreeze_resnet_top
from preprocessing import (
    build_training_records, compute_class_weights,
    duplicate_aware_train_validation_split, make_dataset,
    set_global_seed, write_split_manifest, load_test_records,
)
from evaluate import evaluate as run_evaluation
from gradcam import load_xray, make_gradcam_heatmap, overlay_heatmap
from plot_history import plot_history


def save_run_config(output_dir: Path, model_name: str, lr: float, epochs: int,
                    class_weights: dict, extra: dict | None = None) -> None:
    config = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
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


def get_callbacks(output_dir: Path) -> list:
    return [
        tf.keras.callbacks.ModelCheckpoint(
            output_dir / "best_model.keras", monitor="val_loss",
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(output_dir / "training_history.csv"),
    ]


def train_model(model: tf.keras.Model, train_ds, val_ds, class_weights: dict,
                epochs: int, output_dir: Path) -> tf.keras.callbacks.History:
    output_dir.mkdir(parents=True, exist_ok=True)
    callbacks = get_callbacks(output_dir)
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs,
        callbacks=callbacks, class_weight=class_weights, verbose=2,
    )
    model.save(output_dir / "last_model.keras")
    (output_dir / "history.json").write_text(
        json.dumps({k: [float(v) for v in vals] for k, vals in history.history.items()}, indent=2),
        encoding="utf-8",
    )
    return history


def run_gradcam_gallery(model: tf.keras.Model, output_dir: Path, n_per_class: int = 4) -> None:
    """Generate Grad-CAM visualizations for sample test images."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    test_root = DATASET_ROOT / "test"
    samples = []
    for cls in ("NORMAL", "PNEUMONIA"):
        cls_dir = test_root / cls
        images = sorted(cls_dir.iterdir())[:n_per_class]
        for img_path in images:
            samples.append((cls, img_path))

    fig, axes = plt.subplots(len(samples), 3, figsize=(12, 4 * len(samples)))
    if len(samples) == 1:
        axes = [axes]

    for idx, (true_class, img_path) in enumerate(samples):
        original, batch = load_xray(str(img_path), IMAGE_SIZE)
        heatmap, prob, layer_name = make_gradcam_heatmap(batch, model)
        colored, overlay_img = overlay_heatmap(original, heatmap)
        pred_class = "PNEUMONIA" if prob >= 0.5 else "NORMAL"
        confidence = prob if pred_class == "PNEUMONIA" else 1 - prob

        axes[idx][0].imshow(original)
        axes[idx][0].set_title(f"True: {true_class}", fontsize=10)
        axes[idx][0].axis("off")

        axes[idx][1].imshow(colored)
        axes[idx][1].set_title(f"Grad-CAM ({layer_name})", fontsize=10)
        axes[idx][1].axis("off")

        axes[idx][2].imshow(overlay_img)
        axes[idx][2].set_title(f"Pred: {pred_class} ({confidence:.1%})", fontsize=10)
        axes[idx][2].axis("off")

        # Save individual overlay
        from PIL import Image
        Image.fromarray(overlay_img).save(output_dir / f"{true_class}_{img_path.stem}_overlay.png")

    fig.suptitle("Grad-CAM Explainability Gallery", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "gradcam_gallery.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grad-CAM gallery saved to {output_dir}")


def create_comparison(results_dirs: dict[str, Path], output_dir: Path) -> None:
    """Load test_metrics.json from each model and produce a comparison table and chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_keys = ["accuracy", "precision_pneumonia", "recall_sensitivity_pneumonia",
                    "specificity_normal", "f1_pneumonia", "roc_auc"]
    display_names = ["Accuracy", "Precision", "Recall/Sensitivity",
                     "Specificity", "F1 Score", "ROC-AUC"]

    all_metrics = {}
    for name, rdir in results_dirs.items():
        metrics_file = rdir / "evaluation" / "test_metrics.json"
        if metrics_file.exists():
            data = json.loads(metrics_file.read_text(encoding="utf-8"))
            all_metrics[name] = {k: data.get(k, 0.0) for k in metrics_keys}

    if not all_metrics:
        print("  No evaluation results found for comparison.")
        return

    # Save comparison table as JSON
    (output_dir / "comparison_metrics.json").write_text(
        json.dumps(all_metrics, indent=2), encoding="utf-8"
    )

    # Create comparison bar chart
    model_names = list(all_metrics.keys())
    x = np.arange(len(display_names))
    width = 0.8 / len(model_names)

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
    ax.set_title("Model Comparison — Test Set Performance")
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(display_names, rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=180)
    plt.close(fig)

    # Find best model
    best_name = max(all_metrics, key=lambda n: all_metrics[n]["f1_pneumonia"])
    (output_dir / "best_model.txt").write_text(
        f"Best model by F1: {best_name}\n" + json.dumps(all_metrics[best_name], indent=2),
        encoding="utf-8",
    )
    print(f"  Comparison saved to {output_dir}")
    print(f"  Best model by F1: {best_name}")
    return best_name


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    set_global_seed(SEED)
    print("=" * 72)
    print("MEDICAL IMAGE CLASSIFICATION — FULL PIPELINE")
    print(f"  TensorFlow {tf.__version__}  •  Python {platform.python_version()}")
    print(f"  GPUs: {tf.config.list_physical_devices('GPU') or 'None (CPU only)'}")
    print(f"  Dataset: {DATASET_ROOT}")
    print("=" * 72)

    # ── Data preparation ─────────────────────────────────────────────────
    print("\n[1/8] Preparing data splits...")
    records = build_training_records(DATASET_ROOT)
    train_records, val_records = duplicate_aware_train_validation_split(records, 0.15, SEED)
    class_weights = compute_class_weights(train_records)
    print(f"  Train: {len(train_records)} | Val: {len(val_records)} | Class weights: {class_weights}")

    train_ds = make_dataset(train_records, IMAGE_SIZE, BATCH_SIZE, training=True, seed=SEED)
    val_ds   = make_dataset(val_records,   IMAGE_SIZE, BATCH_SIZE, training=False, seed=SEED)

    # ── Step 1: Baseline CNN ─────────────────────────────────────────────
    cnn_dir = RESULTS_DIR / "baseline_cnn" / "full_2026_09_10"
    print(f"\n[2/8] Training Baseline CNN ({CNN_EPOCHS} epochs)...")
    cnn_model = build_baseline_cnn(IMAGE_SIZE)
    cnn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                 tf.keras.metrics.AUC(name="auc")],
    )
    save_run_config(cnn_dir, "baseline_cnn", 1e-3, CNN_EPOCHS, class_weights)
    train_model(cnn_model, train_ds, val_ds, class_weights, CNN_EPOCHS, cnn_dir)

    print("\n[3/8] Evaluating Baseline CNN on test set...")
    cnn_eval_dir = cnn_dir / "evaluation"
    run_evaluation(cnn_dir / "best_model.keras", DATASET_ROOT, cnn_eval_dir, IMAGE_SIZE, BATCH_SIZE)

    # Plot training curves
    plot_history(cnn_dir / "training_history.csv", cnn_eval_dir)

    # ── Step 2: ResNet50 Frozen ──────────────────────────────────────────
    rn_frozen_dir = RESULTS_DIR / "resnet50" / "frozen_2026_09_10"
    print(f"\n[4/8] Training ResNet50 frozen head ({RN_FROZEN_EPOCHS} epochs)...")
    rn_model = build_resnet50(IMAGE_SIZE)
    rn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                 tf.keras.metrics.AUC(name="auc")],
    )
    save_run_config(rn_frozen_dir, "resnet50_frozen", 1e-4, RN_FROZEN_EPOCHS, class_weights,
                    {"base_trainable": False, "pretrained_weights": "imagenet"})
    # Write split manifest for ResNet50
    manifest = rn_frozen_dir / "training_validation_manifest.csv"
    if manifest.exists():
        manifest.unlink()
    write_split_manifest(train_records, "train", manifest)
    write_split_manifest(val_records, "validation", manifest)
    train_model(rn_model, train_ds, val_ds, class_weights, RN_FROZEN_EPOCHS, rn_frozen_dir)

    print("\n[5/8] Evaluating Frozen ResNet50 on test set...")
    rn_frozen_eval_dir = rn_frozen_dir / "evaluation"
    run_evaluation(rn_frozen_dir / "best_model.keras", DATASET_ROOT, rn_frozen_eval_dir, IMAGE_SIZE, BATCH_SIZE)
    plot_history(rn_frozen_dir / "training_history.csv", rn_frozen_eval_dir)

    # ── Step 3: Fine-tune ResNet50 ───────────────────────────────────────
    rn_ft_dir = RESULTS_DIR / "resnet50" / "finetuned_2026_09_10"
    print(f"\n[6/8] Fine-tuning ResNet50 ({RN_FINETUNE_EPOCHS} epochs, lr=1e-5)...")
    # Load the best frozen model and unfreeze top layers
    best_frozen = tf.keras.models.load_model(rn_frozen_dir / "best_model.keras")
    unfreeze_resnet_top(best_frozen, trainable_layers=30)
    best_frozen.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                 tf.keras.metrics.AUC(name="auc")],
    )
    save_run_config(rn_ft_dir, "resnet50_finetuned", 1e-5, RN_FINETUNE_EPOCHS, class_weights,
                    {"base_trainable_layers": 30, "fine_tuned_from": str(rn_frozen_dir / "best_model.keras"),
                     "batchnorm_frozen": True})
    train_model(best_frozen, train_ds, val_ds, class_weights, RN_FINETUNE_EPOCHS, rn_ft_dir)

    print("\n[7/8] Evaluating Fine-tuned ResNet50 on test set...")
    rn_ft_eval_dir = rn_ft_dir / "evaluation"
    run_evaluation(rn_ft_dir / "best_model.keras", DATASET_ROOT, rn_ft_eval_dir, IMAGE_SIZE, BATCH_SIZE)
    plot_history(rn_ft_dir / "training_history.csv", rn_ft_eval_dir)

    # ── Step 4: Compare and select best ──────────────────────────────────
    print("\n[8/8] Comparing models and generating Grad-CAM...")
    comparison_dir = RESULTS_DIR / "comparison"
    best_name = create_comparison({
        "Baseline CNN": cnn_dir,
        "ResNet50 (Frozen)": rn_frozen_dir,
        "ResNet50 (Fine-tuned)": rn_ft_dir,
    }, comparison_dir)

    # Copy best model for Streamlit
    best_model_map = {
        "Baseline CNN": cnn_dir / "best_model.keras",
        "ResNet50 (Frozen)": rn_frozen_dir / "best_model.keras",
        "ResNet50 (Fine-tuned)": rn_ft_dir / "best_model.keras",
    }
    final_model_dir = MODELS_DIR / "resnet50"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = final_model_dir / "final_model.keras"
    if best_name and best_name in best_model_map:
        shutil.copy2(best_model_map[best_name], final_model_path)
        print(f"  Copied {best_name} → {final_model_path}")
    else:
        # Default to fine-tuned ResNet50
        shutil.copy2(rn_ft_dir / "best_model.keras", final_model_path)
        print(f"  Copied fine-tuned ResNet50 → {final_model_path}")

    # Grad-CAM gallery using the best model
    best_model = tf.keras.models.load_model(final_model_path)
    gradcam_dir = RESULTS_DIR / "gradcam"
    run_gradcam_gallery(best_model, gradcam_dir, n_per_class=4)

    print("\n" + "=" * 72)
    print("PIPELINE COMPLETE")
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Final model: {final_model_path}")
    print(f"  Grad-CAM gallery: {gradcam_dir / 'gradcam_gallery.png'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
