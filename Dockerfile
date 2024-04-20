FROM python:3.10-slim

RUN apt-get update
RUN apt-get install -y sudo curl gnupg wget
RUN useradd -ms /bin/bash app
RUN echo "app ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
USER app


RUN echo "deb http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && sudo curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
RUN sudo apt-get update
RUN wget 'http://storage.googleapis.com/tensorflow-serving-apt/pool/tensorflow-model-server-2.8.0/t/tensorflow-model-server/tensorflow-model-server_2.8.0_all.deb'
RUN sudo dpkg -i tensorflow-model-server_2.8.0_all.deb

RUN pip install --upgrade pip
RUN pip install matplotlib tensorflow requests tensorflow-serving-api==2.8.0