# Fase 1 – Modelo predictivo

Modelo de regresión que estima el **precio (AUD) de un vehículo** en el mercado australiano a partir de las características de su anuncio.

## Contenido de la carpeta

| Archivo | Descripción |
|---|---|
| `fase1_modelo_predictivo.ipynb` | Notebook ejecutable: EDA, preparación, comparación de modelos, ajuste, evaluación e interpretación |
| `features.py` | Limpieza sin estado de las columnas crudas (la usan el notebook y el modelo guardado) |
| `models/model_v1.joblib` | Pipeline completo entrenado (limpieza + preprocesamiento + modelo) |
| `models/model_v1_metadata.json` | Versión, fecha, hiperparámetros, métricas y versiones de librerías |
| `models/sample_input.json` | Registros crudos de ejemplo con su precio real (para pruebas en fases 2 y 3) |

## Cómo ejecutar

Desde la raíz del repositorio:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
cd fase-1
jupyter notebook fase1_modelo_predictivo.ipynb
```

O ejecutarlo completo desde la terminal (tarda ~4 minutos):

```bash
cd fase-1
jupyter nbconvert --to notebook --execute --inplace fase1_modelo_predictivo.ipynb
```

El notebook lee los datos de `../data/raw/australian_vehicle_prices.csv` y **regenera** `models/`. Con `random_state=42` los resultados son reproducibles.

## Datos

- **Fuente:** [Australian Vehicle Prices – Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/australian-vehicle-prices)
- 16.734 registros × 19 columnas; objetivo `Price`.
- Se eliminan 54 filas cuyo **objetivo** no es utilizable: 50 con `"POA"` (precio a consultar), 3 nulas y 1 con un precio evidentemente erróneo (< 500 AUD). Resultado: 16.680 filas.
- Requisito de nulos: `Car/Suv` (0,17 %) y `BodyType` (1,69 %) tienen entre 0,1 % y 2 % de valores faltantes. Además, varias columnas usan `"-"` como marcador de dato faltante (~10 % en `Engine`, `FuelConsumption`, `Doors` y `Seats`).

## Metodología

1. **División train/test 80/20 al inicio**, estratificada por deciles de precio. El test se usa una sola vez.
2. **EDA solo sobre train.**
3. **Construcción de variables** (`features.clean_raw`, reglas fijas fila a fila): se extraen números de textos (`"4 cyl, 2.2 L"` → 2,2 L y 4 cilindros; `"8.7 L / 100 km"` → 8,7), el estado a partir de `Location` y el color exterior. Se descartan `Title` (redundante, casi un identificador) y `Car/Suv` (texto sucio que duplica `BodyType`).
4. **Pipeline de scikit-learn:**
   - Numéricas: imputación por mediana + indicador de faltante → `StandardScaler`
   - Categóricas de baja cardinalidad: imputación `"Desconocido"` → `OneHotEncoder` (agrupa las categorías con < 20 registros)
   - `Brand` y `Model` (alta cardinalidad): `TargetEncoder` con *cross-fitting*
   - Objetivo transformado con `log1p` (`TransformedTargetRegressor`), porque el precio tiene una asimetría muy fuerte (9,3 → 0,16 en escala logarítmica)
5. **Comparación con CV 5-fold:** Dummy, Ridge, RandomForest y HistGradientBoosting.
6. **Ajuste de hiperparámetros** de HistGradientBoosting con `RandomizedSearchCV` (25 combinaciones, 5-fold).
7. **Evaluación final en test**, análisis por segmento de precio e importancia por permutación.

### Prevención de fuga de información
- El test se separa antes de cualquier análisis o ajuste.
- Todo lo que aprende de los datos (imputación, escalado, codificación) está dentro del pipeline y se ajusta solo con los datos de entrenamiento de cada pliegue.
- El `TargetEncoder` usa *cross-fitting*: ninguna fila se codifica con su propio precio.
- La limpieza previa al pipeline es sin estado: no depende de estadísticas del conjunto.

## Resultados

**Validación cruzada (train, 5-fold)**

| Modelo | MAE (AUD) | RMSE (AUD) | R² | MAPE |
|---|---|---|---|---|
| Dummy (mediana) | 18.611 | 38.326 | −0,05 | 61,9 % |
| Ridge | 8.274 | 23.305 | 0,62 | 20,2 % |
| RandomForest | 5.765 | 18.602 | 0,75 | 14,3 % |
| HistGradientBoosting | 5.940 | 17.502 | 0,78 | 14,8 % |
| **HistGradientBoosting ajustado** | **5.370** | – | – | – |

**Test (3.336 vehículos no vistos)**

| Métrica | Modelo final | Línea base |
|---|---|---|
| MAE | **4.930 AUD** | 18.478 AUD |
| Error absoluto mediano | **2.573 AUD** | 11.685 AUD |
| RMSE | 11.333 AUD | 35.306 AUD |
| R² | **0,892** | −0,05 |
| MAPE | **13,2 %** | 62,6 % |

- El 82 % de las predicciones tiene un error ≤ 20 %.
- Variables más importantes (permutación): `Year`, `Kilometres`, `Model`, `Brand`, `Engine`, `DriveType`.
- Mayor error relativo en los extremos (< 15k y > 100k AUD, ~20 %) que en el rango medio (~10–12 %).

## Uso del modelo guardado

El modelo recibe **registros crudos** con el mismo formato del CSV. Las columnas faltantes se tratan como nulas y se imputan.

```python
import sys, joblib, pandas as pd
sys.path.insert(0, "fase-1")          # features.py debe ser importable
model = joblib.load("fase-1/models/model_v1.joblib")

car = pd.DataFrame([{
    "Brand": "Toyota", "Year": 2019, "Model": "Corolla", "UsedOrNew": "USED",
    "Transmission": "Automatic", "Engine": "4 cyl, 1.8 L", "DriveType": "Front",
    "FuelType": "Hybrid", "Kilometres": "65000", "BodyType": "Hatchback",
}])
print(model.predict(car))   # precio estimado en AUD
```

> El modelo se serializó con scikit-learn 1.9.1: para cargarlo, usa la misma versión (ver `requirements.txt`).
