FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TF_CPP_MIN_LOG_LEVEL=2 PORT=8000

WORKDIR /app
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY src/ src/
COPY api/ api/
COPY ui/ ui/
COPY data/ data/
COPY models/bean_model.h5 models/metrics.json models/

WORKDIR /app/api
EXPOSE 8000
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
