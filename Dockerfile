FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

WORKDIR /workdir

RUN pip install --no-cache-dir numpy pandas torch lightning tensorboard pyarrow fastparquet