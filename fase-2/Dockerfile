# Imagen oficial de Python 3.14
FROM python:3.14-slim

# Configuración del entorno
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Directorio de trabajo
WORKDIR /app

# Instalar las dependencias
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Comando predeterminado
CMD ["python", "train.py", "--help"]