FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y tmux

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt