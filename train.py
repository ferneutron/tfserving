import os
import tempfile
import tensorflow as tf 
import pandas as pd 

TRAIN_DATASET = "train.csv"
TARGET_NAME = "target"

BUFFER_SIZE = 1000
BATCH_SIZE = 32
EPOCHS = 10

MODEL_DIR = tempfile.gettempdir()   # Directory where the trained model will be exported
MODEL_VERSION = "1" 

def load_data():
    df = pd.read_csv(TRAIN_DATASET)
    target = df.pop(TARGET_NAME)

    dataset = tf.data.Dataset.from_tensor_slices((df, target))
    batches = dataset.shuffle(buffer_size=BUFFER_SIZE).batch(batch_size=BATCH_SIZE)

    return batches


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])

    return model 


if __name__ == '__main__':
    batches = load_data()

    model = build_model()
    model.summary()

    model.fit(batches, epochs=EPOCHS, batch_size=BATCH_SIZE)

    export_path = os.path.join(MODEL_DIR, MODEL_VERSION)
    model.export(export_path)