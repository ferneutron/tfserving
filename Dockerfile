FROM python:3.10-slim

RUN apt-get update
RUN apt-get install -y sudo curl gnupg
RUN useradd -ms /bin/bash app
RUN echo "app ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
USER app

RUN pip install --pip upgrade
RUN pip install matplotlib tensorflow requests
RUN echo "deb http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && sudo curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
RUN sudo apt update