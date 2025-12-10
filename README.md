# 🪨📄✂️ Rock–Paper–Scissors Image Classification with CNNs

This project implements an image classification system using **Convolutional Neural Networks (CNNs)** with TensorFlow/Keras to classify hand-gesture images into **Rock, Paper, Scissors & Background**.  
The project includes a full training pipeline, dataset integration via Kaggle API, evaluation utilities, and both Notebook & Python script execution options.

---

## Overview
Deep learning project demonstrating an end-to-end machine-learning workflow:

- Dataset import & preprocessing via **Kaggle API**
- Convolutional Neural Network built with **TensorFlow/Keras**
- Model training, validation, early stopping & checkpointing
- Data visualization, misclassification analysis & evaluation metrics
- Ready-to-run using Notebook or `main.pyt` script

---

## Features

### ✨ Core Capabilities
- Automated dataset download
- TF.Data pipeline: batching, caching, prefetching
- 3-layer CNN + dropout regularization
- Early stopping + best-model checkpointing
- Visual metrics and error analysis

### 🧪 Training Outputs
- Accuracy & loss plots
- Saved models:
  - `best_rps_model.keras`
  - `rps_cnn_final.keras`
- Misclassified image preview

---

## Installation

### Requirements
- Python **3.10+**
- Install dependencies:
```bash
pip install -r requirements.txt
```

### Run the Project

**Option A – Notebook**
```bash
jupyter notebook main.ipynb
```

**Option B – Python Script**
```bash
python main.pyt
```

---

## Dataset

- Source: Kaggle – Rock Paper Scissors dataset  
  https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors

### Setup
Place `kaggle.json` in:
```
~/.kaggle/kaggle.json
```

The project automatically downloads & extracts images.

---

## Model Architecture

| Layer | Description |
|---|---|
| Input | 150×150×3 |
| Conv2D ×3 + MaxPool | feature extraction |
| Dense + Dropout | classification head |
| Output | 4-class Softmax |

~225K trainable parameters.

---

## Configuration

```python
NUM_EPOCHS = 30
BATCH_SIZE = 32
IMAGE_SIZE = (150,150)
EARLY_STOPPING_PATIENCE = 5
VALIDATION_SPLIT = 0.2
```

Optimizer: **Adam 0.001**  
Loss: **SparseCategoricalCrossentropy**

---

## Usage Examples

### Train the Model
```bash
python main.pyt
```

### Load for Inference
```python
model = tf.keras.models.load_model('rps_cnn_final.keras')
pred = model.predict(img_batch)
```

---

## Project Structure

```
Image-Classification-with-CNNs/
├── main.ipynb
├── main.pyt
├── requirements.txt
├── README.md
└── data/
    └── rps_data/... (auto-generated)
```

---

## Future Enhancements
- [ ] Data augmentation
- [ ] Transfer learning (ResNet/MobileNet)
- [ ] TensorBoard integration
- [ ] Confusion matrix reporting
- [ ] Deploy inference API (Flask/FastAPI)

---

## Author
Developed as a deep-learning project for educational use and experimentation with CNN image classification.

---

**Happy Training 🤖🔥**
