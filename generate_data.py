import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


N_SAMPLES = 5000       # Define the number of samples (data points)
N_FEATURES = 10        # Define the number of features
N_CLASSES = 2          # Set the number of classes (binary classification = 2)
TEST_SIZE = 0.2        # Sie of test set
SEED = 42

def generate_data():

    x, y = make_classification(
        n_samples=N_SAMPLES, 
        n_features=N_FEATURES, 
        n_classes=N_CLASSES)

    x_train, x_test, y_train, y_test = train_test_split(
        x, 
        y, 
        test_size=TEST_SIZE, 
        random_state=SEED)

    df_train = pd.DataFrame(x_train, columns=[f"feature_{i}" for i in range(N_FEATURES)])
    df_train["target"] = y_train

    df_test = pd.DataFrame(x_test, columns=[f"feature_{i}" for i in range(N_FEATURES)])
    df_test["target"] = y_test

    print(f"Train: {df_train.shape}")
    print(f"Test: {df_test.shape}")

    df_train.to_csv("train.csv", index=False)
    df_test.to_csv("test.csv", index=False)

if __name__ == '__main__':
    generate_data()