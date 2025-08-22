# Binary_classificator_via_ResNet50
In this project, I train a ResNet50 via PyTorch to determine whether a person is present in a photo, using a dataset of approximately 10,000 images.


## Quick Start
```bash
# Recommended Python version 3.10+
pip install torch torchvision timm albumentations pandas scikit-learn pillow tqdm matplotlib
jupyter notebook Human_or_not_classification.ipynb
```

Run the notebook cells sequentially. Exit by stopping the notebook kernel.

---

## What Happens in the Notebook

### 1. Data Loading
- CSV files (`train.csv`, `valid.csv`) contain image filenames (`id`) and labels (`target_people` — 0 or 1).  
- Images are loaded from `train/` and `valid/` directories.  
- A custom PyTorch Dataset (`CustomDataset`) handles image reading and transformation.  
- `DataLoader` creates batches for training and validation.

### 2. Data Augmentation
**Training transforms:**
- Horizontal and vertical flips  
- Rotation ±25°  
- Random resized crop (scale 0.8–1.0 or 0.9 for experiments)  
- Color jitter (brightness, contrast, saturation, hue)  
- Normalization to ImageNet mean/std  
- Conversion to PyTorch tensor  

**Validation/test transforms:**
- Resize to target image size (128×128 or 224×224)  
- Normalization  
- Conversion to tensor  

### 3. Model Selection
- Main model: **ResNet50** with the last fully connected layer modified to output **2 classes**.  
- Other supported models: EfficientNet-B0, MobileNet V2/V3, RegNet, ShuffleNet V2.  

### 4. Training Pipeline
- Loss: `CrossEntropyLoss`  
- Optimizer: `Adam`  
- Optional scheduler: `StepLR`  
- Training supports **subset experiments** (fraction of the training set) to speed up hyperparameter search.  
- Early stopping applied if loss increases significantly.  
- Model weights are saved after each epoch with **validation accuracy and loss in the filename**.  

**Hyperparameter experiments in the notebook:**
- Batch sizes: 16, 32, 64, 128, 224  
- Learning rates: from 0.05 down to 0.0000001  
- Image sizes: 128×128 and 224×224  
- Each combination trained and weights saved for analysis.

### 5. Validation
- The best model is loaded (`load_resnet_best_percent`)  
- Computes:
  - `Validation Accuracy`  
  - `ROC AUC Score`  
- Also extracts **probabilities for class 1** (person) for inspection.  

**Results achieved:**
```
Validation Accuracy: 0.897
ROC AUC Score: 0.955
```
- This indicates **high accuracy** and excellent class separation.  

### 6. Test Predictions
- Test images are loaded from `test/` folder.  
- Model predicts **class probabilities** for each image.  
- Predictions are written to `updated_sample_submission.csv` in the column `target_people`.

---

## Pipeline Overview
1. Load CSV files and images → prepare Dataloaders  
2. Apply augmentations to training images  
3. Initialize ResNet50 (pretrained, last layer 2 outputs)  
4. Train with different batch sizes, learning rates, and image sizes  
5. Save model weights after each epoch  
6. Evaluate best model on validation set → accuracy & ROC AUC  
7. Make predictions on test set → save probabilities in CSV  

---

## Tips & Observations
- Using **subset of training data** accelerates hyperparameter search.  
- Higher batch sizes and smaller learning rates generally improve stability.  
- Image augmentation improves generalization.  
- ROC AUC > 0.95 shows the model is very confident even in borderline cases.  

---

## Dependencies
- Python 3.10+  
- `torch`, `torchvision`, `timm`  
- `albumentations`  
- `pandas`, `numpy`  
- `scikit-learn`  
- `Pillow`  
- `tqdm`  
- `matplotlib` (optional for visualization)

