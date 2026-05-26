FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai

# Install apt packages
RUN apt-get update && \
    apt-get install -y \
        ca-certificates wget curl git vim unzip ffmpeg tzdata \
        gcc g++ make \
    && ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Asia/Shanghai" > /etc/timezone \
    && \
    apt-get autoclean

# Set timezone
RUN dpkg-reconfigure -f noninteractive tzdata

# Install uv and create a Python environment managed by uv.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:/root/.local/bin:${PATH}"

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    uv python install 3.10.12 && \
    uv venv --python 3.10.12 /opt/venv

# Install core CUDA runtime dependencies before the project dependencies.
# onnxruntime-gpu requires cudnn9, which is installed by pytorch 2.4.1,
# do not use an image with cudnn8
COPY pyproject.toml /opt/pyproject.toml
RUN uv pip install --python /opt/venv/bin/python \
        torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu124 && \
    uv cache clean

# COPY code
COPY . /workspace/chatterbox
# Install code
RUN cd /workspace/chatterbox && \
    uv pip install --python /opt/venv/bin/python . && \
    uv cache clean

# Set working directory
WORKDIR /workspace/chatterbox

# Set entrypoint
ENTRYPOINT ["/bin/bash", "-c", "PYTHONPATH=/workspace/chatterbox python service/main.py --config_path service/config.py"]
