import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Define the number of samples (data points)
n_samples = 5000

# Define the number of features
n_features = 10

# Set the number of classes (binary classification = 2)
n_classes = 2

# Generate the data and target labels
x, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

df_train = pd.DataFrame(x_train, columns=[f"feature_{i}" for i in range(n_features)])
df_train["target"] = y_train

df_test = pd.DataFrame(x_test, columns=[f"feature_{i}" for i in range(n_features)])
df_test["target"] = y_test

print(f"Train: {df_train.shape}")
print(f"Test: {df_test.shape}")

# df_train.to_csv("train.csv", index=False)
# df_test.to_csv("test.csv", index=False)

