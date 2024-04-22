import pandas as pd 
import tensorflow as tf 

EPOCHS = 10
BATCH_SIZE = 32
BUFFER_SIZE = 1000

def load_dataset():
    df = pd.read_csv("train.csv")
    target = df.pop("target")
    dataset = tf.data.Dataset.from_tensor_slices((df, target))
    batches = dataset.shuffle(buffer_size=BUFFER_SIZE).batch(batch_size=BATCH_SIZE)

    return batches


def get_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1)
        ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return model


if __name__ == "__main__":
    batches = load_dataset()

    model = get_model()
    model.summary()
    model.fit(batches, epochs=15, batch_size=BATCH_SIZE)