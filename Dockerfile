FROM nvidia/cuda:11.3.1-base-ubuntu20.04

ENV PYTHON_VERSION=3.9

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -qq update \
    && apt-get -qq install --no-install-recommends \
    git \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    ffmpeg \
    libopencv-dev \
    python3-opencv \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 && \
    ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    ln -s -f /usr/bin/pip3 /usr/bin/pip

RUN pip install --upgrade pip

RUN pip install opencv-python

RUN pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

RUN pip install  "git+https://github.com/ai-forever/Kandinsky-2.git"

RUN pip install git+https://github.com/openai/CLIP.git

WORKDIR /home

COPY app.py /home





