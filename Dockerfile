FROM python:3.10-slim

# Combine updates and installations of system packages 
RUN apt-get update && \
    apt-get install -y sudo curl gnupg wget && \
    useradd -ms /bin/bash app && \
    echo "app ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

USER app

RUN echo "deb http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
    curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add - && \
    sudo apt-get update

RUN sudo wget 'http://storage.googleapis.com/tensorflow-serving-apt/pool/tensorflow-model-server-2.8.0/t/tensorflow-model-server/tensorflow-model-server_2.8.0_all.deb' && \
    sudo dpkg -i tensorflow-model-server_2.8.0_all.deb

# Install Python packages together
RUN sudo pip install --upgrade pip
RUN sudo pip install requests \
                     pandas \
                     scikit-learn \
                     protobuf==3.20.* \
                     tensorflow \
                     tensorflow-serving-api==2.8.0  \
                     tensorflow_serving_api