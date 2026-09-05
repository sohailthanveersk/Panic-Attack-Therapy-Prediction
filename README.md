# Panic Attack Therapy Prediction Using Machine Learning

## Project Overview

Developed a machine learning classification system to predict therapy requirements based on panic-attack-related physiological, behavioral, lifestyle, and symptom features.

The project performs data preprocessing, exploratory data analysis, feature encoding, model training, and performance evaluation using Support Vector Machine (SVM) and XGBoost classifiers.

> **Disclaimer:** This is an academic/portfolio project and is not a medical diagnostic or treatment system.

## Objective

The objective of this project is to analyze panic-attack-related factors and build a classification model capable of predicting whether therapy may be required.

## Project Workflow

Dataset
↓
Data Preprocessing
↓
Exploratory Data Analysis
↓
Feature Engineering
↓
Train-Test Split
↓
SVM & XGBoost
↓
Model Evaluation
↓
Therapy Prediction

## Models
- Support Vector Machine (SVM)
- XGBoost

## Evaluation Metrics
Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

## Reported Results
| Model | Accuracy |
|---|---:|
| SVM | 52.9% |
| XGBoost | 94.1% |

## Results

### Exploratory Data Analysis
  ![EDA](Results/EDA_Plots.png)

### Confusion Matrices
![Confusion Matrices](Results/Confusion_Matrices.png)

### Model Comparison
![Model Comparison](Results/Model_Comparison.jpg)

## Technologies
Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Joblib.

## Project Structure
```text
Panic-Attack-Therapy-Prediction/
├── Dataset/
│   └── README.md
├── Results/
│   ├── EDA_Plots.jpg
│   ├── Confusion_Matrices.jpg
│   └── Model_Comparison.jpg
├── panic_attack_therapy_prediction.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Run
```bash
pip install -r requirements.txt
python panic_attack_therapy_prediction.py
```

## Future Work
Larger datasets, cross-validation, hyperparameter tuning, additional classifiers, feature-importance analysis, and application deployment.

## Disclaimer
For educational and portfolio purposes only; not for medical diagnosis or treatment decisions.
