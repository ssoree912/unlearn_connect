# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG APT_FORCE_IPV4=false

# Base tools and shared runtime libraries.
RUN printf 'Acquire::http::Proxy "false";\nAcquire::https::Proxy "false";\n' > /etc/apt/apt.conf.d/99no-proxy && \
    apt-get update -o Acquire::ForceIPv4=${APT_FORCE_IPV4} && \
    apt-get install -y --no-install-recommends \
      ca-certificates curl wget bzip2 \
      unzip procps build-essential \
      git screen \
      git-lfs \
      ffmpeg \
      libsm6 libxext6 libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Miniforge only. Conda env creation is left to container runtime.
RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/mf.sh && \
    bash /tmp/mf.sh -b -p /opt/conda && \
    /opt/conda/bin/conda --version && \
    rm /tmp/mf.sh

ENV PATH=/opt/conda/bin:$PATH
SHELL ["bash", "-lc"]

CMD ["bash"]
