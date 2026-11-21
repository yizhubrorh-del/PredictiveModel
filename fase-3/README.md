# Fase 3 — API REST

**Proyecto:** Predicción de precios de vehículos en Australia  
**Asignatura:** Modelos y Simulación de Sistemas I  
**Repositorio:** [PredictiveModel](https://github.com/yizhubrorh-del/PredictiveModel)

## 1. Objetivo y alcance

Esta fase publica el modelo de la Fase 2 como un servicio REST con tres endpoints:

| Método | Ruta | Función |
|---|---|---|
| `GET` | `/health` | Estado del servicio y del modelo por defecto |
| `POST` | `/predict` | Precio estimado (AUD) de uno o varios vehículos |
| `POST` | `/train` | Entrena y guarda una versión nueva del modelo |

Se implementa con **FastAPI**, que valida las entradas con Pydantic y genera documentación interactiva en `/docs`.

**Sin código duplicado:** la API importa `train.py`, `src/` y los modelos versionados (`models/v1`, `models/v2`) directamente desde `fase-2/`. `app/config.py` agrega `fase-2` al `sys.path`. Esto es necesario porque los `.joblib` referencian `src.features.clean_raw`. Las pruebas también reutilizan el generador de datos sintéticos y el modelo rápido de `fase-2/tests/test_aplicacion.py`.

El monitoreo pertenece a la Fase 4 y **no se implementa aquí**. Para que la Fase 4 pueda ampliar esta API sin copiarla, `create_app` acepta dos parámetros opcionales: `active_version`, que elige la versión por defecto de `/predict`, y `on_prediction`, que se llama tras cada predicción y puede devolver `ids_prediccion`. Sin ellos, el comportamiento es el descrito aquí.

## 2. Estructura

```text
fase-3/
├── app/
│   ├── __init__.py
│   ├── config.py        # variables de entorno y acceso a fase-2
│   ├── schemas.py       # validación de entradas/salidas (Pydantic)
│   ├── registry.py      # carga y caché de versiones del modelo
│   └── main.py          # endpoints y manejo de errores
├── tests/
│   └── test_api.py      # 30 pruebas pytest
├── client.py            # cliente de prueba (requests)
├── requirements.txt
├── Dockerfile
├── Dockerfile.dockerignore
└── README.md
```

## 3. Instalación y ejecución local

PowerShell, desde la raíz del repositorio:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r fase-3/requirements.txt
cd fase-3
uvicorn app.main:app --port 8000
```

`fase-3/requirements.txt` incluye las dependencias de la Fase 2 (`-r ../fase-2/requirements.txt`) y añade `fastapi==0.165.0`, `uvicorn==0.54.0`, `pydantic==2.14.4`, `requests==2.34.2` (cliente) y `httpx2==2.16.2` (TestClient de las pruebas).

- Documentación interactiva: <http://localhost:8000/docs>
- Esquema OpenAPI: <http://localhost:8000/openapi.json>

### Configuración

| Variable | Valor por defecto | Uso |
|---|---|---|
| `MODELS_DIR` | `fase-2/models` | Carpeta de versiones del modelo |
| `DATA_PATH` | `data/raw/australian_vehicle_prices.csv` | Dataset que usa `/train` |
| `DEFAULT_MODEL_VERSION` | `v1` | Versión que usa `/predict` si no se indica otra |
| `FASE2_DIR` | `fase-2` | Ubicación del código de la Fase 2 |

## 4. Endpoints

### `GET /health`

```json
{
  "status": "ok",
  "modelo_por_defecto": "v1",
  "modelo_cargado": true,
  "versiones_disponibles": ["v1", "v2"],
  "dataset_disponible": true,
  "detalle": null
}
```

Responde `503` con `"status": "degradado"` si el modelo por defecto no existe o no se puede cargar.

### `POST /predict[?version=v2]`

El cuerpo es una lista de 1 a 1000 vehículos, con los **mismos nombres de columna que el CSV original** (sin `Price`):

```json
{
  "vehicles": [
    {
      "Brand": "Toyota", "Model": "Corolla", "Year": 2018,
      "UsedOrNew": "USED", "Transmission": "Automatic",
      "Engine": "4 cyl, 1.8 L", "DriveType": "Front", "FuelType": "Unleaded",
      "FuelConsumption": "6.1 L / 100 km", "Kilometres": 85000,
      "ColourExtInt": "White / Black", "Location": "Parramatta, NSW",
      "CylindersinEngine": "4 cyl", "BodyType": "Hatchback",
      "Doors": "5 Doors", "Seats": "5 Seats"
    }
  ]
}
```

Respuesta:

```json
{ "version": "v1", "moneda": "AUD", "n": 1, "predicciones": [28819.56] }
```

**Reglas de validación**

- `Brand`, `Model` y `Year` son **obligatorios**. `Year` debe ser un entero entre 1940 y el año actual + 1.
- Los demás campos son opcionales. Si faltan o son `null`, los imputa el pipeline, igual que los nulos en `predict.py`.
- `Kilometres` debe ser un número entre 0 y 5.000.000.
- `Doors`, `Seats`, `CylindersinEngine` y `FuelConsumption` aceptan el texto del dataset (`"4 Doors"`) o un número (`4`), que se convierte al mismo formato.
- No se admiten campos desconocidos (p. ej. `Price`), textos vacíos ni listas vacías.
- `version` solo admite letras, números, `-`, `_` y `.`, lo que impide rutas como `../`.

### `POST /train`

```json
{ "version": "v3", "seed": 42, "test_size": 0.2, "source_commit": null }
```

Llama a `train()` de `fase-2/train.py` con el dataset configurado en el servidor. El cliente **no puede indicar rutas**. Es síncrono: con el dataset completo tarda unos 6 s. Responde `201` con métricas del modelo y de la línea base, filas usadas, commit y duración. La versión nueva queda disponible de inmediato en `/predict?version=v3`.

**Evidencia:** con `seed=42` la API reprodujo exactamente las métricas de `v1` (MAE 4.929,86 AUD, R² 0,892, MAPE 13,19 %; 13.344 filas de entrenamiento y 3.336 de prueba).

## 5. Manejo de errores

Todas las respuestas de error tienen la forma `{"error": "...", "detalles": [...]}`. `detalles` solo aparece en errores de validación e indica el campo y el motivo.

| Código | Cuándo |
|---|---|
| `400` | `train.py` rechaza los datos del servidor (p. ej. columnas faltantes) |
| `404` | Versión del modelo inexistente o ruta desconocida |
| `409` | `/train` con una versión que ya existe, o con otro entrenamiento en curso |
| `422` | Entrada inválida: JSON mal formado, campos faltantes o fuera de rango |
| `500` | Error interno; el mensaje es genérico y el detalle queda en el log |
| `503` | `/health` con el modelo no disponible, o `/train` sin dataset |

Ejemplo de `422`:

```json
{
  "error": "Entrada inválida",
  "detalles": [
    { "campo": "vehicles.0.Brand", "mensaje": "Field required" },
    { "campo": "vehicles.0.Year", "mensaje": "Value error, el año no puede ser mayor que 2027" }
  ]
}
```

## 6. Cliente de prueba

Con la API en ejecución, en otra terminal desde `fase-3`:

```powershell
python client.py health
python client.py predict                                  # usa fase-2/data/ejemplo_prediccion.csv
python client.py predict --input <archivo.csv> --version v2
python client.py train --version v-demo-api-001 --seed 42
python client.py demo                                     # health, predict, 422 y 404
python client.py --url http://otra-maquina:8000 health
```

El cliente convierte cada fila del CSV al JSON de `/predict`: las celdas vacías o con `-` se envían como ausentes y se descartan columnas como `Price`. Devuelve código `0` si la petición fue exitosa y `1` si falló.

**Evidencia:** `python client.py demo` contra `uvicorn` devolvió con `v1` las predicciones **61.107,88 · 20.053,32 · 79.467,43 AUD**, idénticas a las de `predict.py` en la Fase 2. Las peticiones inválidas recibieron `422` y `404`.

> Las versiones de prueba conviene nombrarlas `v-demo-*`: `.gitignore` ya excluye `fase-2/models/v-demo-*/`.

## 7. Pruebas automatizadas

Desde `fase-3`:

```powershell
python -m pytest -v
```

Las **30 pruebas** de `tests/test_api.py` usan datos sintéticos y carpetas temporales, así que **no modifican `fase-2/models`**. Cubren:

1. `/health`: estado correcto y estado degradado (`503`) sin modelo.
2. `/predict`: varios vehículos, solo campos obligatorios, equivalencia entre formato numérico y de texto.
3. Validación (`422`): falta `Brand`, año futuro o no numérico, kilómetros negativos, texto vacío, campo extra `Price`, lista vacía, cuerpo vacío y JSON mal formado.
4. Versión inexistente (`404`), versión con ruta maliciosa (`422`) y ruta desconocida (`404`).
5. Error interno (`500`) sin exponer el mensaje original.
6. `/train`: crea una versión utilizable en `/predict` y registra `source_commit`.
7. `/train`: versión duplicada (`409`) sin sobrescribir, parámetros inválidos (`422`), dataset ausente (`503`) y entrenamiento en curso (`409`).
8. El modelo oficial `v1` reproduce por la API las predicciones de la Fase 2.
9. Conversión de CSV a JSON del cliente de prueba.

**Evidencia en Windows:** 30 passed in 4.18s, Python 3.14.7.

## 8. Docker

Como la imagen reutiliza `fase-2`, se construye **desde la raíz del repositorio**. `fase-3/Dockerfile.dockerignore` limita el contexto a lo necesario: `train.py`, `src/`, las pruebas, el CSV de ejemplo y los modelos `v1`/`v2` de la Fase 2, más `fase-3/`. El dataset **no** se copia a la imagen.

PowerShell desde la raíz:

```powershell
$root = (Get-Location).Path
docker build -f fase-3/Dockerfile -t predictivemodel-fase3:1.0 .
```

### 8.1 Solo predicción

```powershell
docker run --rm -p 8000:8000 predictivemodel-fase3:1.0
```

`v1` y `v2` vienen dentro de la imagen. En este modo `/train` responde `503` porque no hay dataset.

### 8.2 Predicción y entrenamiento con persistencia

```powershell
docker run --rm -p 8000:8000 `
  --mount "type=bind,source=$root\data,target=/app/data,readonly" `
  --mount "type=bind,source=$root\fase-2\models,target=/app/fase-2/models" `
  predictivemodel-fase3:1.0
```

El dataset se monta en solo lectura. La carpeta de modelos se monta con escritura para que las versiones creadas con `/train` sobrevivan al contenedor. Dentro de Docker no hay Git, así que conviene enviar `source_commit` en `/train` (`$sha = git rev-parse HEAD`). Si no se envía, `git_commit` queda en `null`, igual que en la Fase 2.

### 8.3 Pruebas dentro del contenedor

```powershell
docker run --rm predictivemodel-fase3:1.0 python -m pytest -v
```

La imagen define un `HEALTHCHECK` contra `/health` y ejecuta **un solo proceso** de uvicorn, porque el bloqueo que evita entrenamientos simultáneos vive en memoria.

**Estado de verificación:** pendiente de ejecutar en Docker. Al redactar esta guía, Docker Desktop no estaba iniciado en el equipo de desarrollo.

## 9. Limitaciones

- `/train` es síncrono y bloquea la petición mientras entrena, unos 6 s con el dataset actual. Con datos más grandes convendría una cola de tareas.
- No hay autenticación. `/train` escribe en disco, por lo que no debe exponerse públicamente sin protección.
- Los modelos `joblib` se cargan solo desde `MODELS_DIR`. No se deben colocar ahí artefactos de terceros.

## 10. Lista de verificación

- [x] API REST con `/health`, `/predict` y `/train`.
- [x] Validación de entradas con mensajes por campo.
- [x] Manejo de errores con formato uniforme (400/404/409/422/500/503).
- [x] Reutilización del código y los modelos de la Fase 2 sin duplicarlos.
- [x] 30 pruebas pytest aprobadas en Windows.
- [x] Cliente de prueba (`client.py`) verificado contra la API en ejecución.
- [x] Dockerfile actualizado para la API.
- [ ] Construcción y ejecución verificadas en Docker.
- [ ] Pull Request a `develop` revisado por otro integrante.
