# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/eyetrack-cache \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /build

COPY requirements-docker.txt ./
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
       --index-url https://download.pytorch.org/whl/cpu \
       torch==2.4.1 torchvision==0.19.1 \
    && python -m pip install -r requirements-docker.txt

COPY unigaze_personalization ./unigaze_personalization
COPY models ./models
COPY scripts/download_models.py ./scripts/download_models.py

# Download the UniGaze weights during the build so classroom startup works
# without contacting Hugging Face.
RUN python scripts/download_models.py


FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/eyetrack-cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_OFFLINE=1 \
    EYETRACK_FACE_LANDMARKER_PATH=/app/models/face_landmarker.task \
    EYETRACK_HOST=0.0.0.0 \
    EYETRACK_PORT=8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local /usr/local
COPY --from=builder /opt/eyetrack-cache /opt/eyetrack-cache

WORKDIR /app

# Only runtime PC/backend, frontend, and eye-tracking assets enter /app.
COPY eye_server.py ./
COPY static ./static
COPY unigaze_personalization ./unigaze_personalization
COPY models ./models
COPY THIRD_PARTY_NOTICES.md ./

RUN mkdir -p /app/data/sessions /app/runs

EXPOSE 8000
VOLUME ["/app/data", "/app/runs"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/api/health', timeout=3).read()" || exit 1

CMD ["python", "eye_server.py"]
