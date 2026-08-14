import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import joblib


# ============================================================
# DATASET LOADING
# ============================================================

DATA_PATH = "data/heart.csv"

df = pd.read_csv(DATA_PATH)

print("========== DATASET INFORMATION ==========")

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nData Types:")
print(df.dtypes)


# ============================================================
# DATA PREPROCESSING
# ============================================================

print("\n========== DATA PREPROCESSING ==========")

X = df.drop("target", axis=1)
y = df["target"]

print("\nFeature shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

print("\n========== TRAIN TEST SPLIT ==========")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training data shape:")
print(X_train.shape)

print("Testing data shape:")
print(X_test.shape)


# ============================================================
# FEATURE SCALING
# ============================================================

print("\n========== FEATURE SCALING ==========")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaled training data shape:")
print(X_train_scaled.shape)

print("Scaled testing data shape:")
print(X_test_scaled.shape)


# ============================================================
# LOGISTIC REGRESSION MODEL
# ============================================================

print("\n========== LOGISTIC REGRESSION ==========")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("Model training completed successfully.")


# ============================================================
# PREDICTION
# ============================================================

print("\n========== PREDICTION ==========")

y_pred = model.predict(X_test_scaled)

print("First 20 actual values:")
print(y_test.values[:20])

print("\nFirst 20 predicted values:")
print(y_pred[:20])


# ============================================================
# MODEL EVALUATION
# ============================================================

print("\n========== MODEL EVALUATION ==========")

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:")
print(accuracy)

print("\nAccuracy Percentage:")
print(f"{accuracy * 100:.2f}%")


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)


# ============================================================
# SAVE MODEL
# ============================================================

print("\n========== SAVING MODEL ==========")

joblib.dump(
    model,
    "outputs/heart_disease_model.joblib"
)

joblib.dump(
    scaler,
    "outputs/heart_disease_scaler.joblib"
)

print("Model saved successfully:")
print("outputs/heart_disease_model.joblib")

print("Scaler saved successfully:")
print("outputs/heart_disease_scaler.joblib")


# ============================================================
# PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("WEEK 4 HEART DISEASE CLASSIFICATION COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nDataset:")
print("Heart Disease Dataset")

print("\nTotal Samples:")
print(len(df))

print("\nTotal Features:")
print(X.shape[1])

print("\nModel:")
print("Logistic Regression")

print("\nAccuracy:")
print(f"{accuracy * 100:.2f}%")

print("\n" + "=" * 60)

# ============================================================
# CONFUSION MATRIX VISUALIZATION
# ============================================================

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

print("\n========== CONFUSION MATRIX VISUALIZATION ==========")

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Disease", "Disease"]
)

disp.plot()

plt.title("Heart Disease - Confusion Matrix")
plt.tight_layout()

plt.savefig("outputs/heart_disease_confusion_matrix.png")
plt.close()

print("Confusion matrix saved to:")
print("outputs/heart_disease_confusion_matrix.png")