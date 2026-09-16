FROM pytorch/pytorch:2.14.0-cuda13.2-cudnn9-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y wget

RUN wget https://github.com/zellij-org/zellij/releases/download/v0.41.2/zellij-x86_64-unknown-linux-musl.tar.gz && \
    tar xzf zellij-x86_64-unknown-linux-musl.tar.gz -C /usr/local/bin && \
    rm zellij-x86_64-unknown-linux-musl.tar.gz

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt