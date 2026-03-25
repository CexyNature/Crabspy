# Crabspy web application image (FastAPI + Jinja2 + HTMX).
# OpenCV and the legacy `crabspy` package are not installed here yet; add when integrated.

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
COPY crabspy /app/crabspy
COPY apps/web /app/apps/web

RUN pip install --no-cache-dir /app && pip install --no-cache-dir /app/apps/web

ENV CRABSPY_DATA_DIR=/data \
    CRABSPY_DATABASE_URL=sqlite:////data/db/project.sqlite

EXPOSE 8000

CMD ["uvicorn", "crabspy_web.app:app", "--app-dir", "/app/apps/web", "--host", "0.0.0.0", "--port", "8000"]
