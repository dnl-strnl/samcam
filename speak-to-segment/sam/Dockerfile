FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# install python.
RUN rm -f /etc/apt/apt.conf.d/docker-clean && \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache && \
    apt-get update && \
    apt-get install -y \
    software-properties-common \
    ca-certificates && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    git \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    libgl1 \
    libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# configure python.
RUN python3.11 -m ensurepip --upgrade && \
    python3.11 -m pip install --upgrade pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# setup app.
WORKDIR /app

# copy litserve files.
COPY sam/server.py ./server.py
COPY sam/config/server/sam3.yaml /app/config/server/sam3.yaml

# setup sam.
RUN git clone https://github.com/facebookresearch/sam3.git /sam3

RUN pip install --no-cache-dir \
  decord \
  einops \
  hydra-core \
  litserve \
  matplotlib \
  omegaconf \
  opencv-python \
  pandas \
  pycocotools \
  psutil \
  scikit-image \
  scikit-learn \
  timm

WORKDIR /sam3
RUN pip install --no-cache-dir -e .

# copy model files.
RUN mkdir -p /sam3/weights
COPY models/sam3/sam3.pt /sam3/weights/sam3.pt
COPY models/sam3/bpe_simple_vocab_16e6.txt.gz /sam3/weights/bpe_simple_vocab_16e6.txt.gz

WORKDIR /app

EXPOSE 11217

CMD ["python3", "server.py", "--config-name", "sam3"]
