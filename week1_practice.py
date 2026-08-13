"""
Week 1 practice set.
This is separate from the Titanic mini-project.

Questions covered:
1. Load CSV with pandas and print first 10 rows.
2. Train/test split using sklearn.
3. Train Linear Regression and check R2 score.
4. Predict house price from area input.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# Small practice dataset based on the Week 1 house-price exercise.
data = {
    "area": [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800],
    "price": [150, 180, 220, 260, 300, 330, 380, 420],
}

df = pd.DataFrame(data)

print("First 10 rows:")
print(df.head(10))

X = df[["area"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
score = r2_score(y_test, predictions)

print("\nR2 score:", score)

area_input = float(input("\nEnter house area in square feet: "))
predicted_price = model.predict([[area_input]])[0]

print(f"Predicted price: {predicted_price:.2f} (same price unit as training data)")
