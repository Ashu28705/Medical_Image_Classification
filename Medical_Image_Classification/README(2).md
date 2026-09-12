# 🩺 MEDVISION AI

## Multimodal Deep Learning-Based Medical Image Classification

**Academic Research Project — Semester 7, Adani University**

MEDVISION AI is a deep learning-based medical image classification platform developed to explore the automated detection and classification of diseases from medical images.

The project uses **Transfer Learning, ResNet50, disease-specific deep learning models, and Grad-CAM explainability** through an interactive **Streamlit** application.

The system is designed as a **modular multimodal platform**, where different diseases can use different image modalities and independently trained classification models.

---

# 📌 Project Title

**Deep Learning-Based Medical Image Classification for Automated Detection and Classification of Diseases**

---

# 🎯 Objectives

The main objectives of MEDVISION AI are:

- To develop a deep learning-based medical image classification system.
- To use **ResNet50 Transfer Learning** for disease classification.
- To support multiple diseases through a modular architecture.
- To support different medical image modalities.
- To provide class probabilities and prediction confidence.
- To implement **Grad-CAM** for explainable AI.
- To develop an interactive Streamlit-based interface.
- To provide a research-oriented platform that can be extended with additional disease-specific models.

---

# 🧠 Current Implementation

The first fully implemented module of MEDVISION AI focuses on:

## Pneumonia Detection from Chest X-Rays

| ComponentDetails |                       |
| ---------------- | --------------------- |
| Image Modality   | Chest X-Ray           |
| Model            | ResNet50              |
| Approach         | Transfer Learning     |
| Classification   | Binary Classification |
| Classes          | NORMAL / PNEUMONIA    |
| Input Size       | 224 × 224             |
| Explainability   | Grad-CAM              |
| Interface        | Streamlit             |

### Pneumonia Pipeline

```text
Chest X-Ray
     ↓
Image Preprocessing
     ↓
ResNet50 Transfer Learning
     ↓
Classification
     ↓
NORMAL / PNEUMONIA
     ↓
Confidence & Probabilities
     ↓
Grad-CAM Explainability

```

The existing Pneumonia module provides an end-to-end workflow for image upload, classification, confidence visualization, and Grad-CAM-based model explanation.

---

# 🚀 Platform Expansion

MEDVISION AI is being developed as a **multimodal disease classification platform** rather than a single-disease classifier.

The proposed architecture is:

```text
                    MEDVISION AI
                         │
                         ▼
                  SELECT DISEASE
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
     Pneumonia      Tuberculosis       Malaria
    Chest X-Ray     Chest X-Ray       Blood Smear
         │               │               │
         ▼               ▼               ▼
    Disease Model   Disease Model   Disease Model
         │               │               │
         └───────────────┼───────────────┘
                         ▼
                     Prediction
                         │
                         ▼
               Confidence & Probabilities
                         │
                         ▼
                   Grad-CAM / XAI

```

Each disease is treated as an independent classification problem with its own:

- Dataset
- Image modality
- Preprocessing pipeline
- Model
- Class labels
- Evaluation results
- Grad-CAM configuration

---

# 🦠 Disease Modules

| DiseaseImage ModalityStatusClassification Approach |                             |                   |                   |
| -------------------------------------------------- | --------------------------- | ----------------- | ----------------- |
| **Pneumonia**                                      | Chest X-Ray                 | ✅ Available       | ResNet50          |
| **Tuberculosis**                                   | Chest X-Ray                 | 🔄 In Development | ResNet50          |
| **Malaria**                                        | Blood-Smear Microscopy      | 📋 Planned        | ResNet50          |
| **Leprosy**                                        | Skin-Lesion Image           | 📋 Planned        | ResNet50          |
| **Cholera**                                        | Microscopy / Pathogen Image | 🧪 Experimental   | Dataset dependent |

## Status Definitions

### ✅ AVAILABLE

The model, inference pipeline, and application integration have been implemented and tested.

### 🔄 IN DEVELOPMENT

Implementation is currently being developed and/or evaluated.

### 📋 PLANNED

The module is part of the proposed platform architecture but has not yet been fully implemented.

### 🧪 EXPERIMENTAL

The module requires further research, dataset validation, or an appropriate image-classification task.

> **Note:** Disease modules are research prototypes and must not be considered clinically validated diagnostic systems.

---

# 🏗️ System Architecture

```text
                     ┌──────────────────────┐
                     │      Streamlit UI    │
                     │      MEDVISION AI    │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │   Disease Selector   │
                     └──────────┬───────────┘
                                │
                  ┌─────────────┼─────────────┐
                  ▼             ▼             ▼
            ┌──────────┐  ┌──────────┐  ┌──────────┐
            │Pneumonia │  │    TB    │  │ Malaria  │
            │  Model   │  │  Model   │  │  Model   │
            └────┬─────┘  └────┬─────┘  └────┬─────┘
                 │             │             │
                 └─────────────┼─────────────┘
                               ▼
                      ┌────────────────┐
                      │   Prediction   │
                      └───────┬────────┘
                              │
                      ┌───────┴────────┐
                      ▼                ▼
              ┌──────────────┐ ┌──────────────┐
              │ Probability  │ │   Grad-CAM   │
              │ & Confidence │ │ Explainable  │
              └──────────────┘ └──────────────┘

```

---

# 🔬 Deep Learning Approach

The primary model architecture is based on **ResNet50 Transfer Learning**.

### Model Pipeline

```text
Input Medical Image
        ↓
224 × 224 × 3
        ↓
ResNet50 Backbone
        ↓
Global Average Pooling
        ↓
Dropout
        ↓
Classification Layer
        ↓
Disease-Specific Classes

```

Transfer learning allows the project to use a pretrained convolutional neural network as a feature extractor and adapt it to medical image classification tasks.

The training process can include:

1. Dataset auditing
2. Data preprocessing
3. Data augmentation
4. Train/validation/test splitting
5. ResNet50 frozen-backbone training
6. Fine-tuning
7. Model evaluation
8. Grad-CAM generation
9. Model deployment through Streamlit

---

# 🔎 Explainable AI — Grad-CAM

MEDVISION AI incorporates **Gradient-weighted Class Activation Mapping (Grad-CAM)** to visualize image regions that contribute to the model's prediction.

### Grad-CAM Workflow

```text
Original Medical Image
        +
Grad-CAM Heatmap
        ↓
Highlighted Image Regions
        ↓
Visual Model Explanation

```

The objective is to make the classification process more interpretable for research and educational purposes.

For the current Pneumonia ResNet50 implementation, the Grad-CAM target layer is:

```text
conv5_block3_3_conv

```

The Grad-CAM interface displays:

- Original image
- Heatmap
- Heatmap overlay

> **Note:** Grad-CAM visualizations represent model attention and should not be interpreted as definitive medical evidence.

---

# 💻 Software Stack

## Programming Language

- Python

## Deep Learning

- TensorFlow
- Keras
- ResNet50

## Machine Learning

- Scikit-learn

## Data Processing

- NumPy
- Pandas
- Pillow

## Visualization

- Matplotlib

## Web Application

- Streamlit

---

# 📂 Project Structure

```text
Medical_Image_Classification/
│
├── Medical_Image_Classification/
│   │
│   ├── app/
│   │   └── app.py
│   │
│   ├── models/
│   │   ├── pneumonia/
│   │   │   └── final_model.keras
│   │   ├── tuberculosis/
│   │   │   └── final_model.keras
│   │   ├── malaria/
│   │   │   └── final_model.keras
│   │   ├── leprosy/
│   │   │   └── final_model.keras
│   │   └── cholera/
│   │       └── final_model.keras
│   │
│   ├── datasets/
│   │   ├── pneumonia/
│   │   ├── tuberculosis/
│   │   ├── malaria/
│   │   ├── leprosy/
│   │   └── cholera/
│   │
│   ├── src/
│   │   ├── diseases/
│   │   ├── common/
│   │   ├── dataset_audit.py
│   │   ├── evaluate.py
│   │   ├── gradcam.py
│   │   ├── models.py
│   │   ├── predict.py
│   │   ├── preprocessing.py
│   │   ├── train.py
│   │   ├── train_resnet50.py
│   │   └── smoke_test.py
│   │
│   ├── results/
│   │   ├── pneumonia/
│   │   ├── tuberculosis/
│   │   ├── malaria/
│   │   ├── leprosy/
│   │   └── cholera/
│   │
│   ├── requirements.txt
│   └── README.md
│
└── Medical_Image_Classification_ResNet50.ipynb

```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Ashu28705/Medical_Image_Classification.git

```

## 2. Navigate to the Project Directory

```bash
cd Medical_Image_Classification

```

## 3. Create a Virtual Environment

```bash
python -m venv venv

```

## 4. Activate the Virtual Environment

### Windows

```bash
venv\Scripts\activate

```

### Linux / macOS

```bash
source venv/bin/activate

```

## 5. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt

```

---

# ▶️ Running the Application

Navigate to the application directory:

```bash
cd Medical_Image_Classification

```

Run the Streamlit application:

```bash
streamlit run app/app.py

```

The application will open in your default web browser.

---

# 🖥️ Application Workflow

```text
1. Open MEDVISION AI
          ↓
2. Select Disease
          ↓
3. View Required Image Modality
          ↓
4. Upload Medical Image
          ↓
5. Run AI Classification
          ↓
6. View Predicted Class
          ↓
7. View Confidence & Probabilities
          ↓
8. View Grad-CAM Explanation

```

---

# 📊 Evaluation Metrics

Each disease-specific model should be evaluated independently using appropriate metrics such as:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

Training and validation curves are also generated where applicable.

### Important

All reported metrics must come from **actual model evaluation**.

No manually created, estimated, or fabricated metrics are used.

Individual prediction confidence values must not be interpreted as overall model accuracy.

---

# 🧪 Research Methodology

The overall research workflow is:

```text
Dataset Collection
       ↓
Dataset Audit
       ↓
Exploratory Data Analysis
       ↓
Preprocessing
       ↓
Data Augmentation
       ↓
Baseline CNN
       ↓
ResNet50 Transfer Learning
       ↓
Fine-Tuning
       ↓
Model Evaluation
       ↓
Grad-CAM Explainability
       ↓
Streamlit Deployment

```

---

# 🔐 Dataset Integrity

The project follows a real-dataset-first approach.

For each disease:

- A verified dataset should be used.
- Dataset classes should be documented.
- Dataset distribution should be analyzed.
- Duplicate images should be checked where feasible.
- Data leakage should be minimized.
- Patient-level separation should be maintained when patient identifiers are available.
- Training and evaluation data should remain separate.

No fake medical datasets should be generated to produce artificial results.

If a suitable dataset is unavailable, the corresponding disease module should remain **Planned** or **Experimental** instead of presenting fabricated results.

---

# 🛡️ Medical Safety Disclaimer

MEDVISION AI is an **academic research and educational prototype**.

The predictions generated by this application:

- are not medical diagnoses;
- are not intended to replace qualified healthcare professionals;
- have not been presented as a certified clinical device;
- should not be used for medical decision-making;
- should be interpreted only within the context of the underlying dataset and model evaluation.

The application is intended to demonstrate the application of deep learning and explainable AI to medical image classification.

---

# 👥 Project Team

### Adani University — Semester 7

| Roll No.Team MemberRole |                      |             |
| ----------------------- | -------------------- | ----------- |
| 010                     | Ashutosh Kamboya     | Team Leader |
| 049                     | Priyanshi Chuadhari  | Team Member |
| 039                     | Meet Katudiya        | Team Member |
| 054                     | Rudri Shukla         | Team Member |
| 018                     | Dhaval Lakum         | Team Member |
| 053                     | Rinkita Ramrakhiyani | Team Member |

---

# 📈 Future Scope

Future development will focus on:

- Completing Tuberculosis classification.
- Adding Malaria blood-smear classification.
- Adding Leprosy skin-lesion classification.
- Investigating suitable image-based Cholera research tasks.
- Improving model evaluation.
- Adding additional explainability techniques.
- Supporting more medical image modalities.
- Improving model robustness and generalization.
- Expanding the platform with additional disease-specific models.

The long-term goal is to develop MEDVISION AI as a **modular research platform for multimodal medical image classification and explainable deep learning**.

---

# 📌 Project Status

| ComponentStatus              |                                        |
| ---------------------------- | -------------------------------------- |
| Dataset Audit                | ✅ Completed for current Pneumonia work |
| EDA                          | ✅ Completed for current Pneumonia work |
| Baseline CNN                 | ✅ Completed                            |
| ResNet50 Transfer Learning   | ✅ Completed                            |
| Pneumonia Classification     | ✅ Available                            |
| Streamlit Interface          | ✅ Available                            |
| Grad-CAM                     | ✅ Available                            |
| Modular Disease Architecture | 🔄 In Development                      |
| Tuberculosis                 | 🔄 In Development                      |
| Malaria                      | 📋 Planned                             |
| Leprosy                      | 📋 Planned                             |
| Cholera                      | 🧪 Experimental                        |
| Multimodal Platform          | 🔄 In Development                      |

---

# ⭐ Key Contribution

The primary contribution of MEDVISION AI is not simply disease classification.

The project focuses on building a **modular and explainable medical image classification framework** in which different diseases can use appropriate image modalities and independently trained deep learning models.

```text
Disease-Specific Data
        +
Disease-Specific Model
        +
Multimodal Input
        +
Explainable AI
        ↓
    MEDVISION AI

```

---

# 🔭 Long-Term Vision

MEDVISION AI aims to demonstrate how deep learning and explainable AI can be combined into a single modular software platform for research in medical image classification.

The architecture is designed so that additional disease-specific models can be integrated without redesigning the complete application.

```text
                 MEDVISION AI
                      │
                      ▼
               Disease Selection
                      │
                      ▼
             Appropriate Modality
                      │
                      ▼
            Disease-Specific Model
                      │
                      ▼
                   Prediction
                      │
                 ┌────┴────┐
                 ▼         ▼
            Confidence  Grad-CAM
                 │         │
                 └────┬────┘
                      ▼
              Explainable Result

```

---

## 📜 License

This project is developed for academic and research purposes.

Refer to the repository license for permitted use and distribution.