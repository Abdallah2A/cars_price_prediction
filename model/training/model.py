from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import joblib


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the dataset from the file
    :return:
    x (pd.DataFrame): train set
    y (pd.Series): label of train set
    """
    x = pd.read_csv("../../data/x.csv")
    y = pd.read_csv("../../data/y.csv").squeeze()

    return x, y


X, y = load_dataset()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("Validation Score:", model.score(X_val, y_val)*100)

# Get model predictions
y_predict = model.predict(X_val)

# Number of features
num_features = X_val.shape[1]

# Create subplots
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))  # 3x3 grid for 7 features
axes = axes.flatten()  # Flatten to easily index subplots

# Loop through each feature
for i, column in enumerate(X_val.columns):
    axes[i].scatter(X_val[column], y_val, color="blue", label="Actual Data", alpha=0.6)
    axes[i].scatter(X_val[column], y_predict, color="red", label="Predicted Data", alpha=0.6)

    axes[i].set_xlabel(column)
    axes[i].set_ylabel("Target Value")
    axes[i].set_title(f"Feature: {column}")
    axes[i].legend()

# Remove empty subplots if there are less than 9
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

joblib.dump(model, "../linear_regression_model.pkl")
print("Model saved successfully!")
