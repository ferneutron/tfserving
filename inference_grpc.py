import grpc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc


def predict_grpc(data, input_name, stub):
    # Create a gRPC request made for prediction
    request = predict_pb2.PredictRequest()

    # Set the name of the model, for this use case it is "model"
    request.model_spec.name = "saved_model"

    # Set which signature is used to format the gRPC query
    # here the default one "serving_default"
    request.model_spec.signature_name = "serving_default"

    # Set the input as the data
    # tf.make_tensor_proto turns a TensorFlow tensor into a Protobuf tensor
    request.inputs[input_name].CopyFrom(tf.make_tensor_proto(data))

    # Send the gRPC request to the TF Server
    result = stub.Predict(request)
    return result


if __name__ == "__main__":

    df = pd.read_csv("test.csv")
    df.drop("target", axis=1, inplace=True)

    # Create a channel that will be connected to the gRPC port of the container
    channel = grpc.insecure_channel("localhost:8500")

    # Create a stub made for prediction
    # This stub will be used to send the gRPCrequest to the TF Server
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    loaded_model = tf.saved_model.load("/tmp/1")
    input_name = list(loaded_model.signatures["serving_default"].structured_input_signature[1].keys())[0]

    grpc_outputs = predict_grpc(df.iloc[0:3].values.tolist(), input_name, stub)
    predictions = np.array([grpc_outputs.outputs['output_0'].float_val])

    print(f"gRPC predictions: {predictions}")