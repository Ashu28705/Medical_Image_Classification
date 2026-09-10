# Deep Learning-Based Medical Image Classification for Automated Detection and Classification of Diseases

## Abstract (draft)

This project develops an educational research prototype for binary automated classification of chest X-ray images into NORMAL and PNEUMONIA classes. It compares a custom convolutional neural network (CNN) baseline with ResNet50 transfer learning and plans to provide Grad-CAM interpretability visualizations and a Streamlit prediction interface. The system is not a certified medical device and is not intended to provide clinical diagnosis. Final comparative results will be added only after the respective models are trained and evaluated on the fixed test set.

## Problem Statement

Reviewing chest X-ray images requires specialist expertise and can be time-consuming. This project investigates whether deep learning can automate binary image classification in a supplied chest X-ray dataset. The scope is limited to research evaluation of image-level labels, rather than clinical diagnosis of individual patients.

## Objectives

1. Audit and preprocess the supplied chest X-ray dataset reproducibly.
2. Implement and evaluate a custom CNN baseline.
3. Implement ResNet50 transfer learning and compare measured performance.
4. Generate Grad-CAM visualizations as model-attention aids.
5. Deliver a Streamlit research prototype using a real saved trained model.

## Dataset Description and Quality Audit

The canonical dataset directory contains 5,856 JPEG images: 5,216 training, 16 supplied validation, and 624 test images. The training distribution is 1,341 NORMAL and 3,875 PNEUMONIA images; hence it is imbalanced toward PNEUMONIA (74.29%). The audit opened every image and found zero unreadable files and zero extremely small images. It found 30 exact duplicate groups, all within a single split; none were exact duplicates across supplied train/validation/test splits.

The supplied validation set has only 16 images, so this project uses a reproducible 15% internal validation subset drawn from the training data for model selection, while retaining the fixed test split for final evaluation. The implementation keeps byte-identical duplicates together in this internal split. Full patient-level separation cannot be verified because filenames contain inconsistent patient information across classes. This is a limitation.

## Initial CNN Result (CNN-001)

One CPU-only baseline epoch was completed and evaluated on the 624-image fixed test set. Its real measured metrics were accuracy 0.625, precision 0.625, recall/sensitivity 1.000, specificity 0.000, F1 0.769, and ROC-AUC 0.817. The thresholded model predicted PNEUMONIA for every test image. This is not an acceptable model result; it is included transparently as an initial baseline measurement, not as evidence of diagnostic performance.

## Results Pending

ResNet50 frozen-head training, fine-tuning assessment, final model comparison, and final Grad-CAM examples remain to be measured. No values are reported for those experiments yet.

## Limitations

The supplied data are imbalanced, metadata do not permit verified complete patient-level separation, and a fixed test score alone does not demonstrate clinical generalisation. Grad-CAM indicates model-contributing regions but cannot establish that a medically correct feature was used. Any future use must include external validation, clinical oversight, bias assessment, and regulatory review.
