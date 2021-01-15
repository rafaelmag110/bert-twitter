FROM tensorflow/tensorflow:latest-gpu
WORKDIR /bert_env

ENV NVIDIA_VISIBLE_DEVICES=2

COPY requirements.txt .
RUN pip install -r requirements.txt