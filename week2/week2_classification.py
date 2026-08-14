from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

TITANIC_FILE = BASE_DIR / "data" / "Titanic-Dataset.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================================================
# MAIN FUNCTION
# =========================================================

def main():

    print("========== WEEK 2 - TITANIC CLASSIFICATION ==========")


    # =====================================================
    # 1. LOAD TITANIC DATASET
    # =====================================================

    if not TITANIC_FILE.exists():

        raise FileNotFoundError(
            f"Titanic dataset not found: {TITANIC_FILE}\n"
            "Please place Titanic-Dataset.csv inside week2/data/"
        )


    df = pd.read_csv(TITANIC_FILE)


    print("\n========== DATASET INFORMATION ==========")

    print("Dataset shape:", df.shape)


    # =====================================================
    # 2. DISPLAY FIRST 5 ROWS
    # =====================================================

    print("\n========== FIRST 5 ROWS ==========")

    print(df.head())


    # =====================================================
    # 3. SELECT IMPORTANT FEATURES
    # =====================================================

    features = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked"
    ]

    target = "Survived"


    data = df[
        features + [target]
    ].copy()


    print("\n========== SELECTED FEATURES ==========")

    print(data.columns.tolist())


    # =====================================================
    # 4. CHECK MISSING VALUES BEFORE CLEANING
    # =====================================================

    print("\n========== MISSING VALUES BEFORE CLEANING ==========")

    print(
        data.isnull().sum()
    )


    # =====================================================
    # 5. HANDLE NUMERICAL MISSING VALUES
    # =====================================================

    if "Age" in data.columns:

        data["Age"] = data["Age"].fillna(
            data["Age"].median()
        )


    if "Fare" in data.columns:

        data["Fare"] = data["Fare"].fillna(
            data["Fare"].median()
        )


    # =====================================================
    # 6. HANDLE CATEGORICAL MISSING VALUES
    # =====================================================

    if "Embarked" in data.columns:

        data["Embarked"] = data["Embarked"].fillna(
            data["Embarked"].mode()[0]
        )


    # =====================================================
    # 7. ENCODE SEX
    # =====================================================

    print("\n========== ENCODING SEX ==========")

    sex_encoder = LabelEncoder()

    data["Sex"] = sex_encoder.fit_transform(
        data["Sex"].astype(str)
    )


    # =====================================================
    # 8. ONE-HOT ENCODE EMBARKED
    # =====================================================

    print("\n========== ENCODING EMBARKED ==========")

    data = pd.get_dummies(
        data,
        columns=["Embarked"],
        dtype=int
    )


    # =====================================================
    # 9. FINAL MISSING VALUE CHECK
    # =====================================================

    print("\n========== MISSING VALUES AFTER CLEANING ==========")

    print(
        data.isnull().sum()
    )


    # =====================================================
    # 10. SEPARATE FEATURES AND TARGET
    # =====================================================

    X = data.drop(
        columns=[target]
    )

    y = data[target]


    print("\n========== FINAL FEATURES ==========")

    print(X.columns.tolist())


    print("\n========== TARGET ==========")

    print("Target:", target)


    # =====================================================
    # 11. TRAIN TEST SPLIT
    # =====================================================

    print("\n========== TRAIN TEST SPLIT ==========")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


    print("Training data:", X_train.shape)
    print("Testing data :", X_test.shape)


    # =====================================================
    # 12. LOGISTIC REGRESSION
    # =====================================================

    print("\n========== LOGISTIC REGRESSION ==========")


    logistic_model = LogisticRegression(
        max_iter=1000
    )


    logistic_model.fit(
        X_train,
        y_train
    )


    logistic_predictions = logistic_model.predict(
        X_test
    )


    logistic_accuracy = accuracy_score(
        y_test,
        logistic_predictions
    )


    print(
        f"Logistic Regression Accuracy: "
        f"{logistic_accuracy:.4f}"
    )


    # =====================================================
    # 13. DECISION TREE
    # =====================================================

    print("\n========== DECISION TREE ==========")


    decision_tree = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )


    decision_tree.fit(
        X_train,
        y_train
    )


    tree_predictions = decision_tree.predict(
        X_test
    )


    tree_accuracy = accuracy_score(
        y_test,
        tree_predictions
    )


    print(
        f"Decision Tree Accuracy: "
        f"{tree_accuracy:.4f}"
    )


    # =====================================================
    # 14. KNN
    # =====================================================

    print("\n========== KNN ==========")


    knn_model = KNeighborsClassifier(
        n_neighbors=5
    )


    knn_model.fit(
        X_train,
        y_train
    )


    knn_predictions = knn_model.predict(
        X_test
    )


    knn_accuracy = accuracy_score(
        y_test,
        knn_predictions
    )


    print(
        f"KNN Accuracy: "
        f"{knn_accuracy:.4f}"
    )


    # =====================================================
    # 15. MODEL COMPARISON
    # =====================================================

    print("\n========== MODEL COMPARISON ==========")


    print(
        f"Logistic Regression: "
        f"{logistic_accuracy:.4f}"
    )


    print(
        f"Decision Tree      : "
        f"{tree_accuracy:.4f}"
    )


    print(
        f"KNN                : "
        f"{knn_accuracy:.4f}"
    )


    # =====================================================
    # 16. FIND BEST CLASSIFICATION MODEL
    # =====================================================

    model_scores = {
        "Logistic Regression": logistic_accuracy,
        "Decision Tree": tree_accuracy,
        "KNN": knn_accuracy
    }


    best_classification_model = max(
        model_scores,
        key=model_scores.get
    )


    best_classification_score = model_scores[
        best_classification_model
    ]


    print(
        f"\nBest Classification Model: "
        f"{best_classification_model}"
    )


    print(
        f"Best Accuracy: "
        f"{best_classification_score:.4f}"
    )


    # =====================================================
    # 17. CONFUSION MATRIX
    # =====================================================

    print("\n========== CONFUSION MATRIX ==========")


    cm = confusion_matrix(
        y_test,
        logistic_predictions
    )


    print(cm)


    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Not Survived",
            "Survived"
        ]
    )


    display.plot()


    plt.title(
        "Titanic Logistic Regression - Confusion Matrix"
    )


    plt.tight_layout()


    confusion_file = (
        OUTPUT_DIR / "titanic_confusion_matrix.png"
    )


    plt.savefig(
        confusion_file,
        dpi=150
    )


    plt.close()


    # =====================================================
    # 18. SAVE MODEL COMPARISON
    # =====================================================

    results = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Decision Tree",
                "KNN"
            ],
            "Accuracy": [
                logistic_accuracy,
                tree_accuracy,
                knn_accuracy
            ]
        }
    )


    results_file = (
        OUTPUT_DIR / "titanic_classification_results.csv"
    )


    results.to_csv(
        results_file,
        index=False
    )


    # =====================================================
    # 19. SAVE CLEANED TITANIC DATA
    # =====================================================

    cleaned_titanic_file = (
        OUTPUT_DIR / "titanic_cleaned.csv"
    )


    data.to_csv(
        cleaned_titanic_file,
        index=False
    )


    # =====================================================
    # 20. FINAL OUTPUT
    # =====================================================

    print(
        "\n========== FILES CREATED =========="
    )


    print(
        "Confusion Matrix:"
    )

    print(confusion_file)


    print(
        "\nClassification Results:"
    )

    print(results_file)


    print(
        "\nCleaned Titanic Dataset:"
    )

    print(cleaned_titanic_file)


    print(
        "\n========== WEEK 2 TITANIC CLASSIFICATION COMPLETED =========="
    )


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":
    main()