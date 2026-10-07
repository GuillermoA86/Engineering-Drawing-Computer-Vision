# Engineering Drawing Symbol Recognition with Computer Vision

## End-to-End Computer Vision Portfolio Project

This project builds an open-source computer vision pipeline for **engineering drawing understanding**, using the public **Symbols in Engineering Drawings (SiED)** dataset.

The industrial use case is the automatic recognition of symbols appearing in **Piping and Instrumentation Diagrams (P&IDs)**. Recognizing symbols can support drawing digitization, engineering document search, asset inventories, quality-control workflows, and downstream CAD/BIM automation.

The public SiED dataset contains **2,432 engineering-symbol instances** extracted from complex P&ID drawings. Each symbol image is represented at 100×100 pixels. The original work describes the dataset as highly imbalanced and provides a train/test protocol. See the original repository and paper in the references below.

> **Scope:** this portfolio project focuses on symbol **classification** from cropped drawing regions. A production detector would add an object-detection stage to locate symbols in a full P&ID before classification.

---

## 1. Business Problem

Engineering drawings contain repeated graphical conventions:

- valves
- instruments
- process components
- connectors
- piping symbols
- measurement symbols

Manual interpretation is expensive and difficult to scale. A computer-vision system can act as a first-pass assistant:

```text
P&ID / engineering drawing
          |
          v
   Symbol localization
          |
          v
     Symbol crop
          |
          v
  CNN / transfer learning
          |
          v
 Predicted symbol + confidence
          |
          v
Structured engineering inventory
```

This project demonstrates the classification stage with a reproducible deep-learning pipeline.

---

## 2. Why This Is a Good Engineering Use Case

Unlike a generic image-classification example, the data comes from real engineering drawings and the target classes represent engineering symbols.

Potential production extensions include:

1. Detect all symbols in a full P&ID.
2. Classify each detected symbol.
3. Read nearby equipment/instrument tags with OCR.
4. Link symbols to text labels.
5. Build a structured drawing representation.
6. Compare revisions automatically.
7. Flag unexpected or missing symbols.
8. Export results to a CAD/BIM or asset-management workflow.

---

## 3. Dataset

### SiED — Symbols in Engineering Drawings

The dataset is published by the authors of the SiED research work:

- 2,432 engineering-symbol instances.
- Extracted from Piping and Instrumentation Diagrams (P&IDs).
- Multiple engineering-symbol classes.
- Strong class imbalance.
- 100×100 pixel symbol images.
- Original split: 80% train / 20% test, with 10% of the training portion used for validation.

Dataset repository:

https://github.com/heyad/Eng_Diagrams

Research paper:

Elyan, E., Moreno, C. G., & Johnston, P. (2020). *Symbols in Engineering Drawings (SiED): An Imbalanced Dataset Benchmarked by Convolutional Neural Networks.*

DOI:

https://doi.org/10.1007/978-3-030-48791-1_16

**Important:** the repository should be checked for its current data-use/licensing terms before redistribution. This portfolio repository does **not** redistribute the original dataset.

---

# 4. End-to-End Pipeline

## Step 1 — Obtain the public dataset

Clone the original dataset repository outside this project or with the helper script:

```bash
python scripts/download_dataset.py
```

The script clones:

```text
https://github.com/heyad/Eng_Diagrams
```

into:

```text
data/raw/Eng_Diagrams
```

The original data remains outside Git tracking through `.gitignore`.

---

## Step 2 — Inspect the dataset

```bash
python scripts/inspect_dataset.py
```

The script recursively searches the downloaded repository for image files and reports:

- image count
- extensions
- dimensions
- candidate class directories
- class frequencies

This is deliberately separated from training so that data assumptions are visible and auditable.

---

## Step 3 — Build a standardized manifest

```bash
python scripts/build_manifest.py
```

The manifest is written to:

```text
data/processed/manifest.csv
```

The standard schema is:

| column | meaning |
|---|---|
| path | image path |
| label | engineering symbol class |
| split | train / validation / test |

The script supports the common folder-per-class layout and can be adapted if the upstream dataset format changes.

---

## Step 4 — Preprocess the symbols

The pipeline applies lightweight computer-vision preprocessing:

- grayscale conversion
- contrast normalization
- resizing
- optional adaptive thresholding
- normalization for the neural network

The goal is not to destroy the line geometry that carries engineering meaning.

---

## Step 5 — Handle class imbalance

The original dataset is strongly imbalanced.

This project therefore compares two approaches:

### Baseline

Standard cross-entropy:

```text
L = CrossEntropy(y, p)
```

### Imbalance-aware training

Class-weighted cross-entropy:

```text
L = WeightedCrossEntropy(y, p)
```

The weight of a class is inversely related to its frequency.

This makes the experiment more relevant to industrial data, where common symbols can dominate the training set while rare symbols may be operationally important.

---

## Step 6 — Train a CNN

The default model is:

```text
ResNet-18
```

using transfer learning from ImageNet.

Only the final classification layer is replaced:

```text
Image
  |
  v
ResNet-18 backbone
  |
  v
feature vector
  |
  v
engineering-symbol classifier
```

The implementation uses PyTorch and torchvision.

---

## Step 7 — Evaluate

The evaluation pipeline reports:

- accuracy
- balanced accuracy
- macro F1
- weighted F1
- confusion matrix
- per-class precision/recall/F1
- top-k accuracy

Macro F1 is particularly important because of the dataset's class imbalance.

Run:

```bash
python src/evaluate.py --checkpoint models/best_model.pt
```

---

## Step 8 — Explain model errors

The project stores:

```text
outputs/confusion_matrix.png
outputs/classification_report.csv
outputs/metrics.json
```

Error analysis should focus on:

- visually similar symbols
- rare classes
- low-confidence predictions
- symbols affected by rotation or line thickness
- annotation ambiguity

---

## Step 9 — Run inference

Single-image inference:

```bash
python src/inference.py \
    --checkpoint models/best_model.pt \
    --image path/to/symbol.png
```

Example output:

```text
Predicted class: gate_valve
Confidence: 0.93

Top predictions:
1. gate_valve       0.93
2. globe_valve      0.04
3. check_valve      0.02
```

---

# 5. Interactive Demo

Launch Streamlit:

```bash
streamlit run src/app.py
```

The application allows an engineer or reviewer to:

1. Upload a cropped engineering symbol.
2. Preview the image.
3. Apply the same preprocessing used during training.
4. Run the trained CNN.
5. Display the predicted class.
6. Display confidence and top-k alternatives.
7. Save a JSON prediction report.

The application is intentionally simple so that the CV model remains the center of the demo.

---

# 6. REST API

Run:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

API documentation:

```text
http://localhost:8000/docs
```

The main endpoint accepts an image and returns structured JSON:

```json
{
  "predicted_class": "gate_valve",
  "confidence": 0.93,
  "top_k": [
    {"class": "gate_valve", "probability": 0.93},
    {"class": "globe_valve", "probability": 0.04}
  ]
}
```

---

# 7. Reproducible Run

## Requirements

Recommended:

- Python 3.10 or 3.11
- CPU is sufficient for experimentation.
- NVIDIA GPU is recommended for faster training.

Create an environment:

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the complete pipeline:

```bash
python scripts/download_dataset.py
python scripts/inspect_dataset.py
python scripts/build_manifest.py
python src/train.py --config configs/resnet18.yaml
python src/evaluate.py --checkpoint models/best_model.pt
streamlit run src/app.py
```

---

# 8. Docker

Build:

```bash
docker build -t engineering-drawing-cv .
```

Run Streamlit:

```bash
docker run --rm -p 8501:8501 engineering-drawing-cv
```

Open:

```text
http://localhost:8501
```

---

# 9. Project Structure

```text
engineering-drawing-cv/
│
├── README.md
├── PROJECT_OVERVIEW.md
├── requirements.txt
├── Dockerfile
├── Makefile
├── .gitignore
│
├── configs/
│   └── resnet18.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── outputs/
│
├── scripts/
│   ├── download_dataset.py
│   ├── inspect_dataset.py
│   └── build_manifest.py
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── api.py
│   └── app.py
│
├── tests/
│   └── test_preprocessing.py
│
└── presentation/
    ├── Engineering_Drawing_CV_Demo.pptx
    └── README.md
```

---

# 10. Technical Decisions

### Why ResNet-18?

It is small enough to run locally while still representing a credible transfer-learning baseline.

### Why transfer learning?

The engineering dataset is relatively small. Starting from ImageNet features reduces the amount of data required to learn generic visual primitives such as edges, contours and local textures.

### Why macro F1?

A model can obtain high accuracy by performing well on common symbols while failing rare classes. Macro F1 gives every class equal importance.

### Why not immediately use an LLM?

The core problem is visual recognition. A CNN provides a measurable, deterministic baseline. An LLM or multimodal model can be added later for reasoning over the structured drawing representation.

### Why classification instead of full object detection?

The public SiED dataset is centered on symbol instances. This makes it ideal for a focused classification project. Production deployment would add a detector such as Faster R-CNN, YOLO or DETR before this classifier.

---

# 11. Production Architecture

A production-grade extension would look like:

```text
                 Engineering Drawing
                         |
                         v
               Image preprocessing
                         |
                         v
                 Object detector
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
       Symbols          Notes         Tables
          |              |              |
          v              v              v
    CNN classifier      OCR        Table parser
          |              |              |
          +--------------+--------------+
                         |
                         v
                 Structured drawing
                         |
                         v
              Engineering validation
                         |
              +----------+----------+
              |                     |
              v                     v
         Asset inventory       QA / revision
```

This creates a natural bridge from computer vision into document AI, OCR, graph construction and engineering automation.

---

# 12. Interview Discussion

This project can support questions such as:

### Computer Vision

- Why is the task classification rather than detection?
- Why ResNet-18?
- Why transfer learning?
- How do you handle grayscale technical drawings?
- Which augmentations are safe for engineering symbols?
- How would rotation affect the model?

### Imbalanced Learning

- Why can accuracy be misleading?
- What is macro F1?
- Why use class-weighted loss?
- Would focal loss help?
- How would you evaluate rare symbols?

### MLOps

- How do you version the dataset?
- How do you reproduce an experiment?
- How do you monitor model drift?
- How would you deploy the model at the edge?
- How would you manage model versions?

### Engineering

- How would you detect symbols in a complete P&ID?
- How would you associate a valve with its tag?
- How would you compare two drawing revisions?
- How would you validate model predictions before an engineering decision?

---

# 13. Future Improvements

1. Add full-image symbol detection.
2. Benchmark YOLO / Faster R-CNN / DETR.
3. Add OCR for engineering tags.
4. Build symbol-to-tag relationships.
5. Add active learning for rare symbols.
6. Add rotation-aware augmentation.
7. Add calibration and confidence thresholds.
8. Export predictions to JSON/CSV.
9. Add experiment tracking.
10. Add CI/CD and automated model tests.
11. Build a graph representation of P&IDs.
12. Add revision-difference detection.

---

# 14. Portfolio Value

This repository demonstrates more than a CNN:

```text
Public data
    ↓
Data acquisition
    ↓
Data inspection
    ↓
Preprocessing
    ↓
Class imbalance
    ↓
Transfer learning
    ↓
Training
    ↓
Evaluation
    ↓
Error analysis
    ↓
Inference
    ↓
FastAPI
    ↓
Streamlit
    ↓
Docker
    ↓
Engineering use case
```

It is therefore designed as a **Data Scientist / Computer Vision / ML Engineer portfolio project**, rather than as an isolated notebook.

---

## References

- SiED dataset repository: https://github.com/heyad/Eng_Diagrams
- SiED paper: https://doi.org/10.1007/978-3-030-48791-1_16
- Deep learning for symbols detection and classification in engineering drawings: https://doi.org/10.1016/j.neunet.2020.05.025

## License

The code in this portfolio project is released under the MIT License.

The original SiED dataset is not redistributed by this repository. Users must follow the dataset authors' current terms of use.
