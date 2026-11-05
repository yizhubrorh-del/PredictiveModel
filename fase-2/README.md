# Fase 2 — Entrenamiento reproducible, predicción y Docker

**Proyecto:** Predicción de precios de vehículos en Australia  
**Asignatura:** Modelos y Simulación de Sistemas I  
**Repositorio:** [PredictiveModel](https://github.com/yizhubrorh-del/PredictiveModel)

## 1. Objetivo y alcance

Esta fase transforma el modelo explorado en la fase 1 en una aplicación Python ejecutable desde línea de comandos, con entrenamiento y predicción independientes, modelos versionados, pruebas automatizadas y un entorno Docker reproducible.

- `train.py`: entrena, evalúa y almacena una versión nueva del modelo.
- `predict.py`: carga una versión existente y genera predicciones sin reentrenar.
- `src/`: contiene la limpieza de variables y la definición del pipeline.
- `models/`: guarda los artefactos de cada versión y sus metadatos.
- `tests/`: verifica comportamientos mediante `pytest`.
- `Dockerfile`: empaqueta el código y sus dependencias; los datos, modelos y resultados se montan desde el equipo anfitrión.

La API REST y el monitoreo pertenecen a fases posteriores y **no se implementan en esta fase**.

## 2. Dataset y enfoque de modelado

**Fuente:** [Australian Vehicle Prices — Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/australian-vehicle-prices)  
**Archivo esperado:** `data/raw/australian_vehicle_prices.csv`, relativo a la raíz del repositorio.  
**Objetivo:** `Price`, expresado en dólares australianos (AUD).

El conjunto original contiene **16.734 registros**. Después de descartar objetivos no numéricos o inferiores a 500 AUD, se utilizan **16.680 registros**. Para las versiones existentes se separaron **13.344 registros de entrenamiento** y **3.336 de prueba** (80/20).

El pipeline de `scikit-learn` integra limpieza determinista de las variables originales, tratamiento de valores faltantes, escalado y codificación de variables categóricas, junto con `HistGradientBoostingRegressor`. La transformación del objetivo utiliza `log1p` durante el ajuste y `expm1` durante la predicción. Las transformaciones que aprenden estadísticas se ajustan solo con el conjunto de entrenamiento para evitar fuga de información.

La línea base predice la mediana del precio del conjunto de entrenamiento. Las métricas se calculan sobre el conjunto de prueba.

## 3. Estructura

```text
PredictiveModel/
├── data/
│   └── raw/
│       └── australian_vehicle_prices.csv
├── fase-1/
├── fase-2/
│   ├── .dockerignore
│   ├── Dockerfile
│   ├── README.md
│   ├── requirements.txt
│   ├── train.py
│   ├── predict.py
│   ├── src/
│   │   ├── __init__.py
│   │   ├── features.py
│   │   └── pipeline.py
│   ├── data/
│   │   └── ejemplo_prediccion.csv
│   ├── models/
│   │   ├── v1/
│   │   │   ├── modelo.joblib
│   │   │   ├── metrics.json
│   │   │   └── metadata.json
│   │   └── v2/
│   │       ├── modelo.joblib
│   │       ├── metrics.json
│   │       └── metadata.json
│   ├── tests/
│   │   └── test_aplicacion.py
│   └── outputs/                     # archivos generados; no se deben versionar
└── README.md
```

> **Integración:** `predict.py` ya está integrado en `develop`. Los archivos de Docker y las pruebas pertenecen a los PR #3 y #4, respectivamente, hasta que sean revisados y fusionados. Para ejecutar todas las instrucciones de esta guía desde `develop`, dichos PR deben estar integrados.

## 4. Requisitos e instalación local

- Python **3.14**.
- Git.
- Docker Desktop con el motor Linux iniciado, únicamente para la ejecución en contenedores.
- Dataset disponible en la ruta descrita en la sección 2.

Comandos **PowerShell**, desde la raíz del repositorio:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r fase-2/requirements.txt
cd fase-2
python train.py --help
python predict.py --help
```

Las dependencias fijadas en `fase-2/requirements.txt` son: `pandas==3.0.6`, `numpy==2.5.3`, `scikit-learn==1.9.1`, `joblib==1.6.0` y `pytest==9.2.0`.

> En Linux/macOS, la activación del entorno virtual es `source .venv/bin/activate`. Ajustar las rutas al sistema operativo correspondiente.

## 5. Entrenamiento local

Desde la carpeta `fase-2`:

```powershell
python train.py `
  --data ../data/raw/australian_vehicle_prices.csv `
  --target Price `
  --output-dir models `
  --version v-demo-local-001 `
  --seed 42 `
  --test-size 0.2
```

`--version` debe ser un identificador **nuevo**. El script rechaza expresamente sobrescribir una carpeta de versión ya existente.

Por cada versión generada, se crean:

```text
models/v-demo-local-001/
├── modelo.joblib     # pipeline ajustado
├── metrics.json      # métricas del modelo y línea base
└── metadata.json     # configuración, origen, versiones y trazabilidad
```

Las versiones oficiales `v1` y `v2` ya están guardadas en el repositorio. No es necesario reentrenarlas para ejecutar predicciones.

### Métricas de las versiones existentes

| Métrica | v1 (semilla 42) | v2 (semilla 43) |
|---|---:|---:|
| MAE del modelo (AUD) | 4.929,86 | 4.835,43 |
| MAE de la línea base (AUD) | 18.477,73 | 18.023,36 |
| R² | 0,892 | 0,883 |
| MAPE | 13,19 % | 13,10 % |

**Importante:** cada versión utilizó una partición de prueba distinta. La tabla documenta los resultados registrados, pero no constituye una comparación controlada para afirmar que una versión es mejor que otra.

## 6. Predicción local

Desde `fase-2`:

```powershell
python predict.py `
  --model models/v1/modelo.joblib `
  --input data/ejemplo_prediccion.csv `
  --output outputs/predicciones.csv

Get-Content outputs/predicciones.csv
```

El CSV de salida contiene una columna llamada **`prediccion`**, una fila por vehículo, en el mismo orden que la entrada, con importes en AUD.

### Esquema de entrada

El archivo CSV debe incluir las columnas obligatorias declaradas en `models/<version>/metadata.json`, entre ellas `Brand`, `Year`, `Model`, `UsedOrNew`, `Transmission`, `Engine`, `DriveType`, `FuelType`, `FuelConsumption`, `Kilometres`, `ColourExtInt`, `Location`, `CylindersinEngine`, `BodyType`, `Doors` y `Seats`.

`Title` y `Car/Suv` son **opcionales**. El script reordena las columnas según el esquema del modelo, completa las opcionales ausentes con nulos e ignora las adicionales. Los valores nulos en columnas presentes son tratados por el pipeline. En cambio, **si falta una columna obligatoria**, devuelve un mensaje de error y código de salida `1`.

Los tres registros del archivo de ejemplo produjeron con `v1` estas predicciones:

| Registro | Predicción (AUD) |
|---|---:|
| Ssangyong Rexton 2022 | 61.107,88 |
| MG MG3 2022 | 20.053,32 |
| BMW 430I 2022 | 79.467,43 |

## 7. Pruebas automatizadas

Desde `fase-2`, con las dependencias instaladas:

```powershell
python -m pytest -v
```

Las cinco pruebas de `tests/test_aplicacion.py` cubren:

1. Entrenamiento y guardado de modelo, métricas y metadatos.
2. Recarga del artefacto y consistencia de predicciones.
3. Procesamiento de valores faltantes.
4. Rechazo de un CSV de entrenamiento al que falta una columna obligatoria.
5. Generación de una segunda versión y rechazo de versiones duplicadas sin sobrescribir los artefactos originales.

Se utilizan datos sintéticos y directorios temporales: las pruebas **no descargan datos** ni modifican las versiones oficiales. El pipeline se ejecuta con menos iteraciones solo durante los tests.

**Evidencia obtenida en Windows:** `5 passed in 15.59s`, Python 3.14.0 y pytest 9.2.0. Esta suite verifica principalmente entrenamiento y versionamiento; la validación de la CLI `predict.py` se comprobó manualmente y sus pruebas automatizadas de integración pueden ampliarse.

## 8. Docker

Todos los comandos de esta sección son para **PowerShell desde la raíz del repositorio**. Primero asegúrate de que los PR de Docker y predicción estén integrados en la rama desde la que construirás la imagen.

### 8.1 Construir la imagen

```powershell
$root = (Get-Location).Path
New-Item -ItemType Directory -Force -Path 'fase-2/outputs' | Out-Null
docker build -t predictivemodel-fase2:1.0 ./fase-2
docker run --rm predictivemodel-fase2:1.0 python train.py --help
```

La construcción instala dependencias y copia los scripts; **no entrena ningún modelo**. `.dockerignore` excluye los directorios de datos, modelos y salidas del contexto de construcción.

### 8.2 Entrenar conservando los artefactos en el equipo

```powershell
docker run --rm `
  --mount "type=bind,source=$root\data,target=/app/data,readonly" `
  --mount "type=bind,source=$root\fase-2\models,target=/app/models" `
  predictivemodel-fase2:1.0 `
  python train.py `
  --data /app/data/raw/australian_vehicle_prices.csv `
  --target Price `
  --output-dir /app/models `
  --version v-demo-docker-001 `
  --seed 42 `
  --test-size 0.2
```

El dataset se monta en **solo lectura**. El directorio de modelos se monta con escritura para conservar `modelo.joblib`, `metrics.json` y `metadata.json` aun cuando se elimine el contenedor con `--rm`. Cambia `--version` si el identificador ya existe.

**Evidencia realizada:** se entrenó `v-docker` dentro del contenedor con MAE 4.929,86 AUD, R² 0,892 y MAPE 13,19 %. Los tres archivos persistieron en Windows.

### 8.3 Predecir con el script dentro de Docker

```powershell
docker run --rm `
  --mount "type=bind,source=$root\fase-2\models,target=/app/models,readonly" `
  --mount "type=bind,source=$root\fase-2\data,target=/app/input,readonly" `
  --mount "type=bind,source=$root\fase-2\outputs,target=/app/outputs" `
  predictivemodel-fase2:1.0 `
  python predict.py `
  --model /app/models/v1/modelo.joblib `
  --input /app/input/ejemplo_prediccion.csv `
  --output /app/outputs/predicciones-docker.csv
```

Si construiste la imagen **antes** de integrar `predict.py`, reconstruye la imagen con el comando de la sección 8.1. La carpeta `outputs` se monta con escritura y las otras dos con solo lectura.

**Estado de verificación:** ya se comprobó dentro de Docker la carga de un modelo con `joblib` y sus tres predicciones mediante una llamada directa a Python. La ejecución de la CLI completa `predict.py` dentro del contenedor queda pendiente de verificar tras la integración y reconstrucción de la imagen.

## 9. Errores esperados y comprobaciones

- **Versión duplicada:** `train.py` rechaza una carpeta existente en `--output-dir` y sale con código `1`.
- **CSV inexistente o inválido:** los scripts muestran errores en la terminal.
- **Columna obligatoria ausente:** `predict.py` rechaza el archivo. Ejemplo validado: `ERROR: Faltan columnas obligatorias en el CSV: Brand` y código `1`.
- **CSV de predicción vacío:** `predict.py` rechaza una entrada sin vehículos.
- **Ruta o montaje incorrectos en Docker:** comprueba `$root`, las carpetas anfitrionas y las rutas internas `/app/...`.

Para cargar los modelos `joblib` se deben usar **únicamente artefactos confiables**; este formato no es seguro para datos arbitrarios de terceros.

## 10. Versionamiento, trazabilidad y Git Flow

Cada directorio de versión contiene `metadata.json` con el identificador, la semilla, las dependencias, la configuración, el origen del dataset, su SHA-256, fechas, cantidades de registros y, cuando es posible, el commit de Git.

**Limitación identificada:** dentro de Docker se excluye `.git` del contexto y la imagen no incorpora Git. Por tanto, el campo `git_commit` de un entrenamiento realizado en el contenedor **puede quedar en `null`** con la implementación actual de `train.py`. Los metadatos de `v1` y `v2` sí registran un commit de origen. Antes del cierre definitivo debe resolverse la trazabilidad de nuevos modelos Docker, por ejemplo mediante un argumento explícito `--source-commit` registrado y validado por el script.

El flujo de trabajo del equipo utiliza:

- `feature/...`: desarrollo aislado por funcionalidad.
- `develop`: integración de Pull Requests revisados por otro integrante.
- `main`: entregas finales después de su validación.

**Evidencias en GitHub:**

- [PR #1 — entrenamiento y versionamiento](https://github.com/yizhubrorh-del/PredictiveModel/pull/1): integrado.
- [PR #2 — predicción](https://github.com/yizhubrorh-del/PredictiveModel/pull/2): aprobado e integrado.
- [PR #3 — Docker](https://github.com/yizhubrorh-del/PredictiveModel/pull/3): pendiente de aprobación e integración al redactar esta guía.
- [PR #4 — pytest](https://github.com/yizhubrorh-del/PredictiveModel/pull/4): pendiente de aprobación e integración al redactar esta guía.

Antes de generar la etiqueta `entrega-2`, se debe comprobar que todas las funciones estén integradas en `develop`, que la ejecución local y en Docker funcione y que se haya actualizado `main` mediante el flujo de revisión acordado.

## 11. Lista de verificación para el cierre

- [x] `train.py` con CLI y artefactos versionados.
- [x] Dos versiones (`v1`, `v2`) con métricas y metadatos.
- [x] `predict.py` con CLI y archivo CSV de ejemplo.
- [x] Construcción de imagen Docker y entrenamiento con persistencia verificados.
- [x] Cinco pruebas de pytest ejecutadas correctamente en Windows.
- [x] Predicción local y validación de error por columna obligatoria comprobadas.
- [ ] Revisar e integrar los PR #3 y #4.
- [ ] Reconstruir la imagen integrada y validar la CLI `predict.py` en Docker.
- [ ] Resolver o documentar formalmente la referencia al commit para entrenamientos Docker.
- [ ] Actualizar el README de la raíz y los integrantes activos.
- [ ] Ejecutar la revisión final, integrar en `main` y crear la etiqueta de entrega.
