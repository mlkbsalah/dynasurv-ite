FROM pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y tmux util-linux && apt-get clean

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt
