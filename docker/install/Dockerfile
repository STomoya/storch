FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

ENV DEBIAN_FRONTEND=noniteractive
RUN apt update -y && \
    apt install -y \
    libopencv-dev \
    git

ARG UID
RUN useradd -l -m -u ${UID} dockeruser
USER ${UID}
ENV PATH=$PATH:/home/dockeruser/.local/bin
