# API REST de la Fase 3.
# La imagen reutiliza el código y los modelos de la Fase 2, por lo que el
# contexto de construcción es la RAÍZ del repositorio:
#   docker build -f fase-3/Dockerfile -t predictivemodel-fase3:1.0 .
# Los archivos incluidos se controlan con fase-3/Dockerfile.dockerignore.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASE2_DIR=/app/fase-2 \
    MODELS_DIR=/app/fase-2/models \
    DATA_PATH=/app/data/raw/australian_vehicle_prices.csv \
    DEFAULT_MODEL_VERSION=v1

WORKDIR /app/fase-3

# Dependencias (fase-3/requirements.txt incluye las de fase-2)
COPY fase-2/requirements.txt /app/fase-2/requirements.txt
COPY fase-3/requirements.txt /app/fase-3/requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

# Código y modelos versionados de la Fase 2 (train.py, src/, v1, v2)
COPY fase-2/ /app/fase-2/
# Código de la API
COPY fase-3/ /app/fase-3/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# Un solo proceso: el bloqueo de /train vive en memoria.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
