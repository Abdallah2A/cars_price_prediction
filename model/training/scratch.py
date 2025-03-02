import numpy as np
import pandas as pd
from typing import Callable, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib


def compute_gradient(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lambda_: float) -> Tuple[np.ndarray, float]:
    """
    Computes the gradient for linear regression with optional L2 regularization (Ridge regression).
    Args:
      x (ndarray (m, n)): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar): model parameters
      lambda_ (scalar): Regularization parameter
    Returns:
      dj_dw (ndarray (n,)): Gradient of the cost w.r.t. w
      dj_db (scalar): Gradient of the cost w.r.t. b
    """
    m, n = x.shape
    dj_dw, dj_db = np.zeros(n), 0
    for i in range(m):
        err = (np.dot(x[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] += err * x[i, j]
        dj_db += err

    # Apply L2 Regularization (Ridge)
    dj_dw += (lambda_ / m) * w

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db


def gradient_descent(x: np.ndarray, y: np.ndarray, w_in: np.ndarray, b_in: float, alpha: float, num_iters: int,
                     gradient_function: Callable, lambda_: float) -> Tuple[np.ndarray, float]:
    """
    Performs gradient descent to fit w,b.
    Args:
      x (ndarray (m, n)): Data, m examples with n features
      y (ndarray (m,)): target values
      w_in (ndarray (n,)): Initial values of model parameters
      b_in (scalar): Initial values of model parameters
      alpha (float): Learning rate
      num_iters (int): Number of iterations to run gradient descent
      gradient_function: Function to call to compute gradients
      lambda_ (float): Regularization parameter
    Returns:
      w (ndarray (n,)): Updated parameter values after gradient descent
      b (scalar): Updated parameter value after gradient descent
    """
    w, b = w_in, b_in
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b, lambda_)
        w -= alpha * dj_dw
        b -= alpha * dj_db

    return w, b


# Load dataset and preprocess it
def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    """
    Load the dataset from CSV files and convert it to NumPy arrays.
    Returns:
      X (np.ndarray): Feature matrix
      y (np.ndarray): Target values
    """
    x = pd.read_csv("../../data/x.csv")

    # Drop the first column if it is an unnamed index column
    if "Unnamed: 0" in x.columns:
        x = x.drop(columns=["Unnamed: 0"])

    X = x.to_numpy()

    y = pd.read_csv("../../data/y.csv").squeeze()
    y = y.values if isinstance(y, pd.Series) else y  # Ensure y is a NumPy array

    return X, y


# Load the dataset
X, y = load_dataset()

# Split dataset into training (80%) and validation (20%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Dynamically initialize parameters
num_features = X_train.shape[1]
b_init = 0.
w_init = np.zeros(num_features)

# Gradient descent settings
iterations = 10000
alpha = 5.0e-2
lambda_ = 1000

# Train the model
w_final, b_final = gradient_descent(X_train, y_train, w_init, b_init, alpha, iterations, compute_gradient, lambda_)

# Predictions on training and validation sets
y_train_pred = np.dot(X_train, w_final) + b_final
y_val_pred = np.dot(X_val, w_final) + b_final

# Calculate accuracy metrics
train_mae = mean_absolute_error(y_train, y_train_pred)
val_mae = mean_absolute_error(y_val, y_val_pred)
train_r2 = r2_score(y_train, y_train_pred)
val_r2 = r2_score(y_val, y_val_pred)

# Print model results
print(f"Final parameters: b = {b_final:.2f}, w = {w_final}")
print(f"Training MAE: {train_mae:.4f}, Validation MAE: {val_mae:.4f}")
print(f"Training R²: {train_r2:.4f}, Validation R²: {val_r2:.4f}")

# Print sample predictions
print("\nSample predictions:")
for i in range(5):  # Print first 5 validation predictions
    print(f"Prediction: {y_val_pred[i]:.2f}, Actual: {y_val[i]:.2f}")

# Save the trained model parameters
joblib.dump({"weights": w_final, "bias": b_final}, "../custom_linear_regression.pkl")
print("Model saved successfully!")
