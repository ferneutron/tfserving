import tensorflow as tf 
import pandas as pd 

def load_dataset():
    df = pd.read_csv("train.csv")
    target = df.pop("target")
    return tf.data.Dataset.from_tensor_slices((dict(df), target))

def build_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(None, 10)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1, activation="relu"),
        ])
    return model


def build_model_v2():

    inputs = {}
    for name in ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4", "feature_5", "feature_6", "feature_7", "feature_8", "feature_9"]:
        inputs[name] = tf.keras.Input(
            shape=(1,), name=name, dtype=tf.float32)

    # Define the model inputs
    # inputs = tf.keras.Input(shape=(10,))

    # Add hidden layers with ReLU activation
    x = tf.keras.layers.Dense(32, activation="relu")(inputs)
    x = tf.keras.layers.Dense(16, activation="relu")(x)

    # Output layer with sigmoid activation for binary classification
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == "__main__":
    dataset = load_dataset()
    model = build_model_v2()

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    model.fit(dataset, epochs=2)