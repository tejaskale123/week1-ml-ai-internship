# Week 2 - Machine Learning & AI Internship

## Projects

Week 2 contains two machine learning projects:

1. House Price Prediction - Regression
2. Titanic Survival Prediction - Classification

---

## Project 1: House Price Prediction

### Dataset
Kaggle House Prices dataset.

### Tasks Completed

- Loaded train and test datasets
- Performed Exploratory Data Analysis
- Checked missing values
- Calculated missing value percentages
- Identified numerical and categorical features
- Performed feature engineering
- Applied One-Hot Encoding
- Handled missing values
- Trained Random Forest model
- Trained Gradient Boosting model
- Compared regression models
- Selected the best model
- Trained the best model on full training data
- Generated house price predictions
- Created Kaggle submission file

### Model Results

| Model | RMSE | MAE | R2 |
|---|---:|---:|---:|
| Random Forest | 29468.62 | 17512.63 | 0.8868 |
| Gradient Boosting | 25744.60 | 15373.75 | 0.9136 |

### Best Model

Gradient Boosting

R2 Score: 0.9136

---

## Project 2: Titanic Survival Classification

### Dataset

Kaggle Titanic Dataset.

### Tasks Completed

- Loaded Titanic dataset
- Selected relevant features
- Checked missing values
- Handled missing Age values
- Handled missing Embarked values
- Encoded Sex
- Applied One-Hot Encoding to Embarked
- Split data into training and testing sets
- Trained Logistic Regression
- Trained Decision Tree
- Trained KNN
- Compared classification models
- Generated confusion matrix
- Saved classification results

### Model Results

| Model | Accuracy |
|---|---:|
| Logistic Regression | 80.45% |
| Decision Tree | 77.65% |
| KNN | 65.92% |

### Best Model

Logistic Regression

Accuracy: 80.45%

---

## Output Files

### House Price Prediction

- train_cleaned.csv
- test_cleaned.csv
- submission.csv
- missing_values report
- SalePrice distribution chart
- SalePrice boxplot

### Titanic Classification

- titanic_cleaned.csv
- titanic_classification_results.csv
- titanic_confusion_matrix.png

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Machine Learning
- Exploratory Data Analysis
- Feature Engineering
- Regression
- Classification