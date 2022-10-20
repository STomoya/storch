FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

ENV DEBIAN_FRONTEND=noniteractive
RUN apt update -y && \
    apt install -y \
    libopencv-dev \
    git

ARG UID
RUN useradd -l -m -u ${UID} dockeruser
USER ${UID}
ENV PATH=$PATH:/home/dockeruser/.local/bin
