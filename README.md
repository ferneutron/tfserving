# Model Deployment with TFServing
This repository contains code to train, export and serve a `Tensorflow` model with `TFServing`. Additionally, this repository provides the installation and configuration of `TFServing` through a `Docker` image.

[![alt text](https://img.youtube.com/vi/video-id/0.jpg)](https://www.youtube.com/watch?v=MCWTS90uGpw)

## How to use it?
It is recommended to follow the step by step described below. However, you can adapt the code to your needs.

### Step 1.
Build the docker image as follows:

```bash
$ docker build -t tfserving:v1 .
```

### Step 2.
Run the container and access it through the shell. 

**Note**: It is recommended that you mount the current directory so that you have access to the python scripts.
If you want to skip mounting the current directory, you will have to modify `Dockerfile` to add `COPY` commands to move scripts from the image build.

```bash
$ docker run -it -v $PWD:/home/app/ tfserving:v1 /bin/bash
```

### Step 3. 
Once you are inside the container, generate the `train.csv` and `test.csv` files, running the `generate_data.py` script.

```bash
$ python -B generate_data.py
```

### Step 4.
Train and export the model.

```bash
$ python -B train.py
```

### Step 5.
Launch `TFServing` server.

```bash
$ tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=saved_model --model_base_path="/tmp"
```

**Note**: Port `8500` will be used for `gRPC` calls while port `8501` will be used for `REST` requests.

### Step 6.
`REST` and `gRPC` requests. 

```bash
$ python -B inference_rest.py
```

```bash
$ python -B inference_grpc.py
```
