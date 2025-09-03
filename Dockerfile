# build a slim image with CPU-friendly deps
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# basic deps for sentence-transformers & PyTorch CPU wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy code
COPY app/ /app/app/
COPY scripts/ /app/scripts/
COPY prompts/ /app/prompts/
COPY config.yaml /app/config.yaml

# create data dir inside container (mounted via volume in compose)
RUN mkdir -p /app/data/chroma

EXPOSE 8520
