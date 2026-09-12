"""Quick smoke test: verify data pipeline, ResNet50 build, one training step,
and that Grad-CAM works end-to-end with a tiny subset. This is a sanity
check before launching the full training pipeline."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("KERAS_HOME", str(PROJECT_ROOT / ".keras_cache"))

import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import tensorflow as tf

from models import build_resnet50
from preprocessing import (
    build_training_records,
    duplicate_aware_train_validation_split,
    make_dataset,
    set_global_seed,
)
from gradcam import load_xray, make_gradcam_heatmap, overlay_heatmap

DATASET_ROOT = PROJECT_ROOT.parent / "chest_xray"
IMAGE_SIZE = 224
BATCH_SIZE = 8
SEED = 42

set_global_seed(SEED)

print("=" * 60)
print("SMOKE TEST — chest_xray pipeline + ResNet50 build")
print("=" * 60)

records = build_training_records(DATASET_ROOT)
print(f"\nTotal training records: {len(records)}")
print(f"NORMAL: {sum(1 for r in records if r['label']==0)}")
print(f"PNEUMONIA: {sum(1 for r in records if r['label']==1)}")

train_recs, val_recs = duplicate_aware_train_validation_split(records, 0.15, SEED)
print(f"After split: train={len(train_recs)}, val={len(val_recs)}")

# Tiny subsets for the smoke test
train_tiny = train_recs[:32]
val_tiny = val_recs[:16]
train_ds = make_dataset(train_tiny, IMAGE_SIZE, BATCH_SIZE, training=True, seed=SEED)
val_ds = make_dataset(val_tiny, IMAGE_SIZE, BATCH_SIZE, training=False, seed=SEED)

batch_x, batch_y = next(iter(train_ds))
print(f"\nBatch shape: {batch_x.shape}, labels: {batch_y.numpy()}")

print("\nBuilding ResNet50 (this loads ImageNet weights from cache)…")
model = build_resnet50(IMAGE_SIZE)
print(f"Total params: {model.count_params():,}")
trainable = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
print(f"Trainable params (frozen base): {trainable:,}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
)

print("\nFitting 1 epoch on tiny subset (32 images)…")
history = model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=2)
print(f"  train_loss={history.history['loss'][-1]:.4f}, "
      f"train_acc={history.history['accuracy'][-1]:.4f}, "
      f"val_loss={history.history['val_loss'][-1]:.4f}, "
      f"val_acc={history.history['val_accuracy'][-1]:.4f}")

print("\nTesting Grad-CAM on a sample image…")
sample = next(iter((DATASET_ROOT / "test" / "PNEUMONIA").iterdir()))
print(f"  sample: {sample.name}")
original, batch = load_xray(str(sample), IMAGE_SIZE)
heatmap, prob, layer = make_gradcam_heatmap(batch, model)
colored, overlay = overlay_heatmap(original, heatmap)
print(f"  heatmap shape: {heatmap.shape}, layer: {layer}")
print(f"  P(PNEUMONIA) = {prob:.4f}")
print(f"  overlay shape: {overlay.shape}")

print("\nSMOKE TEST PASSED")
