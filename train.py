import tensorflow as tf 
import pandas as pd 

def load_dataset():
    df = pd.read_csv("train.csv")
    target = df.pop("target")
    return tf.data.Dataset.from_tensor_slices((df, target))

def build_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(None, 10)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1, activation="relu"),
        ])
    return model


def get_basic_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
        ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return model


def stack_dict(inputs, fun=tf.stack):
    values = []
    for key in sorted(inputs.keys()):
      values.append(tf.cast(inputs[key], tf.float32))

    return fun(values, axis=-1)


def build_model_v2():

    inputs = {}
    for name in ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4", "feature_5", "feature_6", "feature_7", "feature_8", "feature_9"]:
        inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=tf.float32)

    values = []
    for key in sorted(inputs.keys()):
        values.append(tf.cast(inputs[key], tf.float32))

    x = tf.stack
    x = stack_dict(inputs, fun=tf.concat)

    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == "__main__":
    dataset = load_dataset()
    numeric_batches = dataset.shuffle(1000).batch(32)

    model = get_basic_model()
    model.summary()
    model.fit(numeric_batches, epochs=15, batch_size=32)