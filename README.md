# AI-600 Deep Learning: Assignment 0

[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/ahmedjawed/ai-600-assignment-0-datasets)
[![License: CC0](https://img.shields.io/badge/License-CC0-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

This repository contains the coursework for **AI-600 Deep Learning** - Assignment 0, demonstrating fundamental machine learning algorithms implemented from scratch.

---

## 📋 Overview

| Task | Algorithm | Dataset | Objective |
|------|-----------|---------|-----------|
| Task 1 | Linear & Polynomial Regression | Pakistan Used Cars | Predict car prices |
| Task 2 | Naive Bayes Classifier | Heart Disease | Predict heart disease |

---

## 📂 Repository Structure

```
├── assignment_0_task_1.ipynb    # Linear & Polynomial Regression implementation
├── assignment_0_task_2.ipynb    # Naive Bayes Classifier implementation
├── data-a1.csv                  # Pakistan Used Cars dataset (62,303 records)
├── heart-dataset.csv            # Heart Disease dataset (824 records)
├── linear_model.pkl             # Trained regression model
└── README.md
```

---

## 📊 Datasets

### Dataset 1: Pakistan Used Cars (`data-a1.csv`)

**Source:** PakWheels - Pakistan's largest automotive marketplace

| Feature | Description |
|---------|-------------|
| `make` | Manufacturer (Toyota, Honda, Suzuki, etc.) |
| `model` | Specific model name |
| `year` | Manufacturing year |
| `engine` | Engine capacity (cc) |
| `transmission` | Manual/Automatic |
| `fuel` | Fuel type (Petrol, Diesel, Hybrid) |
| `mileage` | Distance traveled (km) |
| `price` | **Target** - Price in PKR |

### Dataset 2: Heart Disease (`heart-dataset.csv`)

**Source:** UCI Machine Learning Repository (modified)

| Feature | Description |
|---------|-------------|
| `age` | Patient age in years |
| `sex` | Male/Female |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `thalach` | Maximum heart rate achieved |
| `target` | **Target** - Disease/No Disease |

---

## 🔬 Task 1: Linear & Polynomial Regression

**Objective:** Predict used car prices based on vehicle features.

### Key Implementations:
- Custom gradient descent from scratch
- Feature scaling with StandardScaler
- One-hot encoding for categorical variables
- Polynomial feature transformation
- Ridge regularization for polynomial regression

### Training Approach:
```python
# Custom Linear Regression with gradient descent
model = CustomLinearRegression(lr=0.1, steps=1500)
model.fit(X_train_scaled, y_train)
```

---

## 🧠 Task 2: Naive Bayes Classification

**Objective:** Predict presence of heart disease using patient clinical data.

### Key Implementations:
- Gaussian Naive Bayes from scratch
- Prior probability calculation
- Gaussian likelihood estimation with smoothing
- Log-probability for numerical stability
- Comparison with sklearn's GaussianNB

### Training Approach:
```python
# Custom Naive Bayes implementation
model = CustomNaiveBayes()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

## 🚀 Quick Start

### Run Locally
```bash
# Clone the repository
git clone https://github.com/ahmedjawedaj/AI-600-Assignment-0.git
cd AI-600-Assignment-0

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Open notebooks
jupyter notebook
```

### Run on Kaggle
```python
# Load datasets from Kaggle
import pandas as pd

cars_df = pd.read_csv('/kaggle/input/ai-600-assignment-0-datasets/data-a1.csv')
heart_df = pd.read_csv('/kaggle/input/ai-600-assignment-0-datasets/heart-dataset.csv')
```

---

## 📦 Dependencies

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- jupyter

---

## 📜 License

This project is released under the **CC0 1.0 Universal** license - free to use for educational and research purposes.

---

## 👤 Author

**Ahmed Jawed**  
MS AI - AI-600 Deep Learning  
Spring 2026
