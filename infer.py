import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import tempfile

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# scale the values to 0.0 to 1.0
train_images = train_images / 255.0
test_images = test_images / 255.0

# reshape for feeding into the model
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)


import json
data = json.dumps({"signature_name": "serving_default", "instances": test_images[0:3].tolist()})
print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))


import requests
headers = {"content-type": "application/json"}
# json_response = requests.post('http://localhost:8501/v1/models/saved_model:predict', data=data, headers=headers)
json_response = requests.post('http://localhost:8501/v1/models/saved_model/versions/1:predict', data=data, headers=headers)

predictions = json.loads(json_response.text)['predictions']