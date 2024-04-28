# Model Deployment with TFServing
This repository contains code to train, export and serve a Tensorflow model with TFServing. Additionally, this repository provides the installation and configuration of TFServing through a Docker image.

## 1. How to use it?
It is recommended to follow the step by step described below. However, you can adapt the code to your needs.

### Step 1.
Build the docker image as follows:

```bash
$ docker build -t tfserving:v1 .
```

### Step 2.
Run the container and access it through the shell. 
Note: It is recommended that you mount the current directory so that you have access to the python scripts.
If you want to skip mounting the current directory, you will have to modify `Dockerfile` to add `COPY` commands to move scripts from the image build.

```bash
$ docker run -it -v $PWD:/home/app/ tfserving:v1 /bin/bash
```

### Step 3. 
Una vez estes dentro del contenedor, genera los archivos `train.csv` y `test.csv`, corriendo el script `generate_data.py`.

```bash
$ python -B generate_data.py
```