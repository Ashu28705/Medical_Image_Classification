# Methodology (current implementation)

## Task

Binary automated chest X-ray image classification: `NORMAL` (0) versus `PNEUMONIA` (1). This is a student research prototype, not a clinical diagnostic system.

## Dataset handling

The canonical source is `C:\DL project\chest_xray`, using only its direct `train`, `val`, and `test` class folders. A nested `chest_xray\chest_xray` copy is intentionally excluded to prevent duplicate-counting and leakage.

Images are decoded to RGB, resized to 224×224, and processed on demand using `tf.data`; the full dataset is not loaded into RAM. ResNet50 applies its own `preprocess_input` function exactly once. The baseline CNN rescales pixels to [0, 1].

## Augmentation

Only the baseline training pipeline uses small rotations (about ±11°), translations (3%), and contrast variation (8%). These model plausible acquisition variation without inventing a new anatomy. No augmentation is used for validation or testing. Horizontal flipping, large rotations, cropping, and aggressive brightness changes are excluded because they can change clinically relevant orientation, anatomy, or imaging characteristics.

## Models

The baseline is a three-block CNN using convolution, batch normalisation, max pooling, global average pooling, dropout, and a sigmoid output. The primary model is ImageNet-pretrained ResNet50 with a frozen base, global average pooling, dropout, and a sigmoid classification head. Fine-tuning will be attempted only after frozen-head evaluation and with a smaller learning rate.

## Evaluation

The fixed test split will produce accuracy, precision, sensitivity/recall, specificity, F1 score, ROC-AUC, a confusion matrix, classification report, and ROC curve. Values are intentionally absent until actual test predictions are generated.
