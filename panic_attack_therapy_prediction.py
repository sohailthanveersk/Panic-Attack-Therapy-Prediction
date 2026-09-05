"""Panic Attack Therapy Prediction Using Machine Learning.
Reconstructed from the original academic project report.
The original dataset and source repository were lost.
"""
import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

DATASET_PATH = "Dataset/Panic_Attack.csv"
TARGET_COLUMN = "Therapy"
RANDOM_STATE = 42

def load_dataset(path=DATASET_PATH):
    df = pd.read_csv(path)
    print("Dataset shape:", df.shape)
    print(df.columns.tolist())
    print(df.isnull().sum())
    return df

def preprocess_data(df):
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0] if df[col].dtype == "object" else df[col].median())
    for col in df.columns:
        if df[col].dtype == "object" and col != TARGET_COLUMN:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df[TARGET_COLUMN].astype(str))
    X = df.drop(columns=[TARGET_COLUMN])
    return X, y

def perform_eda(df):
    plt.figure(figsize=(7,5))
    sns.countplot(x=TARGET_COLUMN, data=df)
    plt.title("Therapy Class Distribution")
    plt.tight_layout()
    plt.show()
    numeric = df.select_dtypes(include=np.number)
    if numeric.shape[1] >= 2:
        plt.figure(figsize=(12,8))
        sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.show()

def evaluate_model(name, y_true, y_pred):
    metrics = (
        accuracy_score(y_true,y_pred)*100,
        precision_score(y_true,y_pred,average="macro",zero_division=0)*100,
        recall_score(y_true,y_pred,average="macro",zero_division=0)*100,
        f1_score(y_true,y_pred,average="macro",zero_division=0)*100
    )
    print(f"\n{name}")
    print(f"Accuracy: {metrics[0]:.2f}%")
    print(f"Precision: {metrics[1]:.2f}%")
    print(f"Recall: {metrics[2]:.2f}%")
    print(f"F1-Score: {metrics[3]:.2f}%")
    print(classification_report(y_true,y_pred,zero_division=0))
    cm = confusion_matrix(y_true,y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout(); plt.show()
    return metrics

def main():
    df = load_dataset()
    perform_eda(df)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=RANDOM_STATE)

    svm = SVC(probability=True, random_state=RANDOM_STATE)
    svm.fit(X_train,y_train)
    svm_metrics = evaluate_model("SVM Classifier",y_test,svm.predict(X_test))

    xgb = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss")
    xgb.fit(X_train,y_train)
    xgb_metrics = evaluate_model("XGBoost Classifier",y_test,xgb.predict(X_test))

    print("\nModel Comparison")
    print(pd.DataFrame([svm_metrics,xgb_metrics],
          index=["SVM","XGBoost"],
          columns=["Accuracy","Precision","Recall","F1-Score"]))

if __name__ == "__main__":
    main()
