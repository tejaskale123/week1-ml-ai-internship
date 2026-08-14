from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

TRAIN_FILE = BASE_DIR / "data" / "train.csv"
TEST_FILE = BASE_DIR / "data" / "test.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

CLEAN_TRAIN_FILE = OUTPUT_DIR / "train_cleaned.csv"
CLEAN_TEST_FILE = OUTPUT_DIR / "test_cleaned.csv"


# =========================================================
# MAIN FUNCTION
# =========================================================

def main():

    print("========== WEEK 2 - DATA CLEANING ==========")

    # -----------------------------------------------------
    # 1. LOAD DATA
    # -----------------------------------------------------

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    print("\n========== ORIGINAL SHAPES ==========")

    print("Train:", train_df.shape)
    print("Test :", test_df.shape)


    # -----------------------------------------------------
    # 2. SEPARATE TARGET
    # -----------------------------------------------------

    y = train_df["SalePrice"]

    train_features = train_df.drop(
        columns=["SalePrice"]
    )

    test_features = test_df.copy()


    # -----------------------------------------------------
    # 3. COMBINE TRAIN + TEST
    # -----------------------------------------------------

    combined = pd.concat(
        [train_features, test_features],
        axis=0,
        ignore_index=True
    )

    print("\n========== COMBINED DATA ==========")

    print("Combined shape:", combined.shape)


    # -----------------------------------------------------
    # 4. CATEGORICAL MISSING VALUES
    # -----------------------------------------------------

    categorical_none_columns = [
        "PoolQC",
        "MiscFeature",
        "Alley",
        "Fence",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "MasVnrType"
    ]

    for column in categorical_none_columns:

        if column in combined.columns:
            combined[column] = combined[column].fillna("None")


    # -----------------------------------------------------
    # 5. NUMERICAL MISSING VALUES
    # -----------------------------------------------------

    numerical_median_columns = [
        "LotFrontage",
        "MasVnrArea",
        "GarageYrBlt"
    ]

    for column in numerical_median_columns:

        if column in combined.columns:
            combined[column] = combined[column].fillna(
                combined[column].median()
            )


    # -----------------------------------------------------
    # 6. ELECTRICAL
    # -----------------------------------------------------

    if "Electrical" in combined.columns:

        combined["Electrical"] = combined["Electrical"].fillna(
            combined["Electrical"].mode()[0]
        )


    # -----------------------------------------------------
    # 7. OTHER CATEGORICAL COLUMNS
    # -----------------------------------------------------

    categorical_columns = combined.select_dtypes(
        include=["object", "str"]
    ).columns

    for column in categorical_columns:

        combined[column] = combined[column].fillna(
            combined[column].mode()[0]
        )


    # -----------------------------------------------------
    # 8. OTHER NUMERICAL COLUMNS
    # -----------------------------------------------------

    numerical_columns = combined.select_dtypes(
        include=["int64", "float64"]
    ).columns

    for column in numerical_columns:

        combined[column] = combined[column].fillna(
            combined[column].median()
        )


    # -----------------------------------------------------
    # 9. FEATURE ENGINEERING
    # -----------------------------------------------------

    print("\n========== FEATURE ENGINEERING ==========")


    # Total bathrooms
    if all(
        col in combined.columns
        for col in [
            "FullBath",
            "HalfBath",
            "BsmtFullBath",
            "BsmtHalfBath"
        ]
    ):

        combined["TotalBathrooms"] = (
            combined["FullBath"]
            + 0.5 * combined["HalfBath"]
            + combined["BsmtFullBath"]
            + 0.5 * combined["BsmtHalfBath"]
        )


    # Total square footage
    if all(
        col in combined.columns
        for col in [
            "TotalBsmtSF",
            "1stFlrSF",
            "2ndFlrSF"
        ]
    ):

        combined["TotalSF"] = (
            combined["TotalBsmtSF"]
            + combined["1stFlrSF"]
            + combined["2ndFlrSF"]
        )


    # Total porch area
    porch_columns = [
        "OpenPorchSF",
        "3SsnPorch",
        "EnclosedPorch",
        "ScreenPorch",
        "WoodDeckSF"
    ]

    if all(
        column in combined.columns
        for column in porch_columns
    ):

        combined["TotalPorchSF"] = (
            combined["OpenPorchSF"]
            + combined["3SsnPorch"]
            + combined["EnclosedPorch"]
            + combined["ScreenPorch"]
            + combined["WoodDeckSF"]
        )


    # House age
    if "YearBuilt" in combined.columns:

        combined["HouseAge"] = (
            combined["YrSold"] - combined["YearBuilt"]
        )


    # Remodel age
    if "YearRemodAdd" in combined.columns:

        combined["RemodAge"] = (
            combined["YrSold"] - combined["YearRemodAdd"]
        )


    # -----------------------------------------------------
    # 10. ONE-HOT ENCODING
    # -----------------------------------------------------

    print("\n========== ONE-HOT ENCODING ==========")

    combined = pd.get_dummies(
        combined,
        drop_first=False,
        dtype=int
    )


    # -----------------------------------------------------
    # 11. HANDLE INFINITE VALUES
    # -----------------------------------------------------

    combined = combined.replace(
        [np.inf, -np.inf],
        np.nan
    )


    # -----------------------------------------------------
    # 12. FINAL MISSING VALUE CHECK
    # -----------------------------------------------------

    remaining_missing = combined.isnull().sum().sum()

    print("\n========== FINAL MISSING VALUES ==========")

    print("Remaining missing values:", remaining_missing)


    # -----------------------------------------------------
    # 13. SPLIT TRAIN + TEST
    # -----------------------------------------------------

    cleaned_train = combined.iloc[
        :len(train_df)
    ].copy()

    cleaned_test = combined.iloc[
        len(train_df):
    ].copy()


    # -----------------------------------------------------
    # 14. SAVE CLEANED DATA
    # -----------------------------------------------------

    cleaned_train["SalePrice"] = y.values

    cleaned_train.to_csv(
        CLEAN_TRAIN_FILE,
        index=False
    )

    cleaned_test.to_csv(
        CLEAN_TEST_FILE,
        index=False
    )


    # -----------------------------------------------------
    # 15. FINAL INFORMATION
    # -----------------------------------------------------

    print("\n========== CLEANING COMPLETED ==========")

    print("Cleaned train shape:", cleaned_train.shape)
    print("Cleaned test shape :", cleaned_test.shape)

    print(
        "\nCleaned train saved to:"
    )

    print(CLEAN_TRAIN_FILE)

    print(
        "\nCleaned test saved to:"
    )

    print(CLEAN_TEST_FILE)

    print(
        "\nWeek 2 Data Cleaning + Feature Engineering completed successfully."
    )


    # =====================================================
    # 16. MODEL TRAINING
    # =====================================================

    print("\n========== MODEL TRAINING ==========")

    X = cleaned_train.drop(
        columns=["SalePrice"]
    )

    y = cleaned_train["SalePrice"]


    # -----------------------------------------------------
    # Train / Validation Split
    # -----------------------------------------------------

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("\nTraining data:", X_train.shape)
    print("Validation data:", X_valid.shape)


    # -----------------------------------------------------
    # RANDOM FOREST
    # -----------------------------------------------------

    print("\n========== RANDOM FOREST ==========")

    random_forest = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    random_forest.fit(
        X_train,
        y_train
    )

    rf_predictions = random_forest.predict(
        X_valid
    )


    # -----------------------------------------------------
    # RANDOM FOREST EVALUATION
    # -----------------------------------------------------

    rf_rmse = np.sqrt(
        mean_squared_error(
            y_valid,
            rf_predictions
        )
    )

    rf_mae = mean_absolute_error(
        y_valid,
        rf_predictions
    )

    rf_r2 = r2_score(
        y_valid,
        rf_predictions
    )

    print("Random Forest RMSE:", rf_rmse)
    print("Random Forest MAE :", rf_mae)
    print("Random Forest R2  :", rf_r2)


    # -----------------------------------------------------
    # GRADIENT BOOSTING
    # -----------------------------------------------------

    print("\n========== GRADIENT BOOSTING ==========")

    gradient_boosting = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    gradient_boosting.fit(
        X_train,
        y_train
    )

    gb_predictions = gradient_boosting.predict(
        X_valid
    )


    # -----------------------------------------------------
    # GRADIENT BOOSTING EVALUATION
    # -----------------------------------------------------

    gb_rmse = np.sqrt(
        mean_squared_error(
            y_valid,
            gb_predictions
        )
    )

    gb_mae = mean_absolute_error(
        y_valid,
        gb_predictions
    )

    gb_r2 = r2_score(
        y_valid,
        gb_predictions
    )

    print("Gradient Boosting RMSE:", gb_rmse)
    print("Gradient Boosting MAE :", gb_mae)
    print("Gradient Boosting R2  :", gb_r2)


    # -----------------------------------------------------
    # MODEL COMPARISON
    # -----------------------------------------------------

    print("\n========== MODEL COMPARISON ==========")

    print(
        f"Random Forest      RMSE: {rf_rmse:.2f} | "
        f"MAE: {rf_mae:.2f} | "
        f"R2: {rf_r2:.4f}"
    )

    print(
        f"Gradient Boosting  RMSE: {gb_rmse:.2f} | "
        f"MAE: {gb_mae:.2f} | "
        f"R2: {gb_r2:.4f}"
    )


    # -----------------------------------------------------
    # SELECT BEST MODEL
    # -----------------------------------------------------

    if gb_rmse < rf_rmse:

        best_model = gradient_boosting
        best_model_name = "Gradient Boosting"

    else:

        best_model = random_forest
        best_model_name = "Random Forest"


    print(
        f"\nBest Model: {best_model_name}"
    )

    # =====================================================
    # 17. TRAIN BEST MODEL ON FULL TRAINING DATA
    # =====================================================

    print("\n========== FINAL MODEL TRAINING ==========")

    best_model.fit(
        X,
        y
    )

    print("Best model trained on full training dataset.")

    # =====================================================
    # 18. PREPARE TEST DATA
    # =====================================================

    X_test = cleaned_test.copy()

    print("\n========== TEST DATA ==========")

    print("Test data shape:", X_test.shape)

    # =====================================================
    # 19. MAKE TEST PREDICTIONS
    # =====================================================

    print("\n========== MAKING PREDICTIONS ==========")

    test_predictions = best_model.predict(
        X_test
    )

    print(
        "Predictions created:",
        len(test_predictions)
    )

    # =====================================================
    # 20. CREATE KAGGLE SUBMISSION
    # =====================================================

    submission = pd.DataFrame(
        {
            "Id": test_df["Id"],
            "SalePrice": test_predictions
        }
    )

    # =====================================================
    # 21. SAVE SUBMISSION FILE
    # =====================================================

    submission_file = OUTPUT_DIR / "submission.csv"

    submission.to_csv(
        submission_file,
        index=False
    )

    print("\n========== SUBMISSION CREATED ==========")

    print(submission.head(10))

    print(
        "\nSubmission shape:",
        submission.shape
    )

    print("\nSubmission saved to:")

    print(submission_file)

# =========================================================
# RUN PROGRAM
# =========================================================
if __name__ == "__main__":
    main()