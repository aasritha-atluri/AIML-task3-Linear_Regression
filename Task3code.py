# Task 3 - Linear Regression (ML Internship)
# Requirements:
# pip install pandas numpy matplotlib scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("Housing.csv") 

print("First 5 rows:\n", df.head())
print("\nInfo:\n", df.info())
print("\nMissing values:\n", df.isnull().sum())

# Preprocessing: Convert categorical to numeric if needed
df = pd.get_dummies(df, drop_first=True)

# Features & target
X = df.drop("price", axis=1)
y = df["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMAE: {mae}")
print(f"MSE: {mse}")
print(f"R²: {r2}")

# Plot regression (only for one feature, e.g., 'area')
if "area" in df.columns:
    plt.scatter(X_test["area"], y_test, color="blue", label="Actual")
    plt.scatter(X_test["area"], y_pred, color="red", label="Predicted")
    plt.xlabel("Area")
    plt.ylabel("Price")
    plt.title("Linear Regression - Area vs Price")
    plt.legend()
    plt.show()

# Coefficients
coeffs = pd.DataFrame(model.coef_, X.columns, columns=["Coefficient"])
print("\nModel Coefficients:\n", coeffs)
