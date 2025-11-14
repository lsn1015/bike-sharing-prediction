import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib

# 1) Load data - Compatible with potential path/encoding differences
csv_path_candidates = [
    r'../data/bike-day.csv',   # Original path
    r'./data/bike-day.csv',   # Alternative spelling
    'data/bike-day.csv'       # Current working directory
]
csv_path = next((p for p in csv_path_candidates if os.path.exists(p)), None)
if csv_path is None:
    raise FileNotFoundError("Cannot find bike-day.csv. Please check the path or place the file in the script directory.")

# Load dataset with encoding fallbacks
try:
    dataset = pd.read_csv(csv_path, encoding='utf-8')
except UnicodeDecodeError:
    try:
        dataset = pd.read_csv(csv_path, encoding='gbk')
    except UnicodeDecodeError:
        dataset = pd.read_csv(csv_path, encoding='latin1')

# 2) Remove unnecessary columns - Skip if columns don't exist
columns_to_remove = [['casual', 'registered'], ['instant', 'dteday']]
for cols in columns_to_remove:
    existing_cols = [c for c in cols if c in dataset.columns]
    if existing_cols:
        dataset = dataset.drop(existing_cols, axis=1)

print("Processed dataset information:")
dataset.info()
print("-" * 30)

# 3) Split data - Preserve variable names from the problem statement
# Features: all columns except 'cnt'
features = [c for c in dataset.columns if c != 'cnt']
# Labels: use 2D DataFrame to match assignment reference shape (n, 1)
labels = dataset[['cnt']]

X, X_test, y, y_test = train_test_split(
    dataset[features], labels,
    test_size=0.33, random_state=42
)

print('X (training features) shape is {}'.format(X.shape))
print('y (training labels) shape is {}'.format(y.shape))
print('-' * 30)
print('X_test (test features) shape is {}'.format(X_test.shape))
print('y_test (test labels) shape is {}'.format(y_test.shape))
print("-" * 30)

# Initialize and train linear regression model
model_lr = linear_model.LinearRegression()
print("Training linear regression model...")
model_lr.fit(X, y)
print("Model training completed.")
print("-" * 30)

# 4) Predictions and visualization
predictions = model_lr.predict(X_test).reshape(-1, 1)

plt.style.use('fivethirtyeight')
plt.figure(figsize=(16, 6))

# Test set: actual values
plt.plot(y_test.values.flatten(), marker=".", label="actual")
# Model predictions
plt.plot(predictions.flatten(), marker=".", label="prediction", color="r")

plt.title("Linear Regression: Actual vs Predicted Values")
plt.xlabel("Sample Index")
plt.ylabel("Bike Sharing Demand")
plt.legend(loc="best")
plt.show()
print("-" * 30)

# 5) Model evaluation - Use flattened arrays for consistent calculation
y_true = y_test.values.flatten()
y_pred = predictions.flatten()
MAE_lr = mean_absolute_error(y_true, y_pred)
MSE_lr = mean_squared_error(y_true, y_pred)
print(f'MAE_lr: {0}, MSE_lr: {1}'.format(MAE_lr, MSE_lr))


model_path = '../models/linear_regression_model.pkl'
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model_lr, model_path)
print(f"Model saved to {model_path}")