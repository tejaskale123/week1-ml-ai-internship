"""
Week 1 - Machine Learning & AI Internship
Mini Project: Titanic Survival Prediction - Data Cleaning Project

Put the Kaggle Titanic CSV at:
    data/Titanic-Dataset.csv
Then run:
    python week1_assignment.py
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "Titanic-Dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
CLEANED_FILE = OUTPUT_DIR / "titanic_cleaned.csv"
AGE_PLOT = OUTPUT_DIR / "age_distribution.png"


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Download the Titanic CSV and save it as data/Titanic-Dataset.csv"
        )

    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Load and explore dataset
    df = pd.read_csv(DATA_FILE)

    print("\n========== FIRST 10 ROWS ==========")
    print(df.head(10))

    print("\n========== DATA INFO ==========")
    df.info()

    print("\n========== DESCRIPTIVE STATISTICS ==========")
    print(df.describe(include="all"))

    print("\n========== MISSING VALUES BEFORE CLEANING ==========")
    print(df.isnull().sum())

    # 2. Handle missing data
    # Age: median is safer when values may contain outliers.
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())

    # Embarked: categorical column, use the most frequent value.
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Cabin has many missing values in the Kaggle Titanic dataset.
    # Keep the project focused on the required Age/Embarked cleaning.
    if "Cabin" in df.columns:
        df["Cabin"] = df["Cabin"].fillna("Unknown")

    # 3. Encode Sex using LabelEncoder
    if "Sex" in df.columns:
        label_encoder = LabelEncoder()
        df["Sex"] = label_encoder.fit_transform(df["Sex"].astype(str))

    # 4. Encode Embarked using OneHotEncoder
    if "Embarked" in df.columns:
        try:
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        except TypeError:
            # Compatibility with older scikit-learn versions.
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse=False
            )

        encoded = encoder.fit_transform(df[["Embarked"]])
        encoded_names = encoder.get_feature_names_out(["Embarked"])
        encoded_df = pd.DataFrame(
            encoded,
            columns=encoded_names,
            index=df.index
        )

        df = pd.concat([df.drop(columns=["Embarked"]), encoded_df], axis=1)

    # 5. Visualize age distribution
    if "Age" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.hist(df["Age"], bins=30)
        plt.title("Titanic Passenger Age Distribution")
        plt.xlabel("Age")
        plt.ylabel("Number of Passengers")
        plt.tight_layout()
        plt.savefig(AGE_PLOT, dpi=150)
        plt.close()

    # 6. Save cleaned dataset
    df.to_csv(CLEANED_FILE, index=False)

    print("\n========== MISSING VALUES AFTER CLEANING ==========")
    print(df.isnull().sum())

    print(f"\nCleaned CSV saved to: {CLEANED_FILE}")
    print(f"Age distribution saved to: {AGE_PLOT}")
    print("\nWeek 1 Titanic data-cleaning project completed successfully.")


if __name__ == "__main__":
    main()
