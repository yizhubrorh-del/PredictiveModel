# API con monitoreo (Fase 4).
# Reutiliza el código y los modelos de las Fases 2 y 3, por lo que el contexto
# de construcción es la RAÍZ del repositorio:
#   docker build -f fase-4/Dockerfile -t predictivemodel-fase4:1.0 .
# Los archivos incluidos se controlan con fase-4/Dockerfile.dockerignore.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASE2_DIR=/app/fase-2 \
    FASE3_DIR=/app/fase-3 \
    MODELS_DIR=/app/fase-2/models \
    DATA_PATH=/app/data/raw/australian_vehicle_prices.csv \
    DEFAULT_MODEL_VERSION=v1 \
    MONITOR_DB=/app/fase-4/logs/monitoreo.db \
    PERFIL_REFERENCIA=/app/fase-4/referencia/perfil_referencia.json

WORKDIR /app/fase-4

# Dependencias (fase-4 -> fase-3 -> fase-2)
COPY fase-2/requirements.txt /app/fase-2/requirements.txt
COPY fase-3/requirements.txt /app/fase-3/requirements.txt
COPY fase-4/requirements.txt /app/fase-4/requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY fase-2/ /app/fase-2/
COPY fase-3/ /app/fase-3/
COPY fase-4/ /app/fase-4/

EXPOSE 8000
# El registro de monitoreo debe sobrevivir al contenedor: montar un volumen aquí.
VOLUME ["/app/fase-4/logs"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# Un solo proceso: el bloqueo de /train vive en memoria.
CMD ["uvicorn", "api_monitoreo:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
