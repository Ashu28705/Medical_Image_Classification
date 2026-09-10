# Experiment Log

| ID | Date | Model | Input | Batch | Learning rate | Epochs | Augmentation | Class weights | Validation / test results | Status |
|---|---|---|---:|---:|---:|---:|---|---|---|---|
| EDA-001 | 2026-09-10 | Dataset audit | Original | N/A | N/A | N/A | None | N/A | 5,856 canonical images; 0 unreadable; 30 exact duplicate groups, none cross-split | Complete |
| CNN-001 | 2026-09-10 | Baseline CNN | 224×224 | 16 | 0.001 | 1 completed (5 requested) | Mild rotation/translation/contrast; train only | NORMAL 1.945; PNEUMONIA 0.673 | Accuracy 0.625; Precision 0.625; Recall 1.000; Specificity 0.000; F1 0.769; ROC-AUC 0.817 | Complete — initial CPU run only |
| RESNET-001 | Not yet run | Frozen ResNet50 head | 224×224 | 16 | 0.0001 | 10 planned | None initially | To be computed | Not yet measured | Pending |

## Dataset-splitting decision

The provided `val` split contains only 16 images (8 per class), which is too small for dependable model selection. Training scripts therefore create a seeded 15% stratified internal validation set from the canonical `train` folders, keeping byte-identical duplicate images in the same subset. The predefined `test` split remains untouched for final evaluation.

Filenames provide inconsistent patient-identification information: many PNEUMONIA names contain `person...`, while NORMAL filenames do not offer a comparable reliable patient ID. A fully patient-level split therefore cannot be verified from the supplied files. This is a documented limitation; no claim of patient-level separation will be made.

## Interpretation of CNN-001

This experiment is an initial one-epoch CPU baseline, not a deployable result. At the 0.5 threshold it predicted PNEUMONIA for every test image (TN=0, FP=234, FN=0, TP=390). Its zero specificity makes the thresholded classifier unacceptable. The ROC-AUC is retained because it is a real score-ranking result, but it does not rescue the model. Further training and threshold analysis are required.
