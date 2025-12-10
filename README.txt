# Rock-Paper-Scissors Image Classification with CNN

## Project Overview
This project implements a Convolutional Neural Network (CNN) to classify hand gestures as Rock, Paper, or Scissors using the Kaggle dataset.

## Dataset
Dataset: [Rock-Paper-Scissors](https://www.kaggle.com/datasets/sartajbhuvaji/rock-paper-scissors)

### Setup Instructions

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2. Download Dataset
1. Visit https://www.kaggle.com/datasets/sartajbhuvaji/rock-paper-scissors
2. Download `rock-paper-scissors.zip`
3. Create a `data` folder in the project directory
4. Place the zip file as `./data/archive.zip`

#### 3. Run the Script
```bash
python main.pyt
```

The script will:
- Extract the dataset from the zip file
- Load images (150x150 pixels) with 80/20 train-validation split
- Build a CNN with 3 Conv layers + MaxPooling
- Train for up to 30 epochs
- Save the best model as `best_rps_model.keras`
- Save the final model as `rps_cnn_final.keras`
- Display up to 6 misclassified examples

## Model Architecture
- Input: 150x150 RGB images
- Rescaling layer: Normalize to [0, 1]
- Conv Block 1: 32 filters → MaxPool
- Conv Block 2: 64 filters → MaxPool
- Conv Block 3: 128 filters → MaxPool
- Dense layers: 128 units (ReLU) + Dropout(0.5)
- Output: 3 classes (Softmax)

## Training Configuration
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Metrics: Accuracy
- Batch size: 32
- Epochs: 30 (with optional EarlyStopping)
- Callbacks: ModelCheckpoint + EarlyStopping

## Configuration Options
Edit `main.pyt` to modify:
- `RUN_FULL_TRAIN`: Set to `True` to run all 30 epochs (default), or `False` to enable EarlyStopping
- `EARLY_STOPPING_PATIENCE`: Number of epochs to wait before early stopping (default: 5)
- `NUM_EPOCHS`: Maximum number of epochs (default: 30)

## Output Files
- `best_rps_model.keras`: Best model saved during training
- `rps_cnn_final.keras`: Final trained model
- Console output: Training history, final accuracy, and misclassified examples

## Results
The model is evaluated on the validation set (20% of data).
Misclassified examples are displayed with predicted and true labels.

## Troubleshooting
- If dataset is not found: Download manually from Kaggle and place at `./data/archive.zip`
- If TensorFlow import fails: Run `pip install -r requirements.txt`
- For GPU acceleration: Install `tensorflow-gpu` instead of `tensorflow`

## Notes
- Images are resized to 150x150 pixels
- Validation split is fixed with seed=123 for reproducibility
- Training can be stopped early if validation loss plateaus
