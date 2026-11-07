"""Limpieza y construcción de variables para el dataset Australian Vehicle Prices.

Todas las funciones de este módulo son *sin estado* (fila a fila): no aprenden
nada de los datos, por lo que pueden aplicarse antes o después de dividir en
train/test sin provocar fuga de información. Todo lo que sí aprende de los
datos (imputación, codificación, escalado) vive dentro del Pipeline de
scikit-learn y se ajusta únicamente con el conjunto de entrenamiento.

El modelo guardado referencia `features.clean_raw`, así que este archivo debe
estar importable (en el mismo directorio o en el PYTHONPATH) para cargarlo.
"""

import numpy as np
import pandas as pd

TARGET = "Price"

# Columnas crudas que el modelo espera recibir (todas excepto el objetivo).
RAW_COLUMNS = [
    "Brand", "Year", "Model", "Car/Suv", "Title", "UsedOrNew", "Transmission",
    "Engine", "DriveType", "FuelType", "FuelConsumption", "Kilometres",
    "ColourExtInt", "Location", "CylindersinEngine", "BodyType", "Doors", "Seats",
]

NUMERIC_FEATURES = [
    "Year", "Kilometres", "EngineLitres", "Cylinders",
    "FuelConsumption", "Doors", "Seats",
]
# Categóricas de baja cardinalidad -> One-Hot
LOW_CARD_FEATURES = [
    "UsedOrNew", "Transmission", "DriveType", "FuelType",
    "BodyType", "State", "ColourExt",
]
# Categóricas de alta cardinalidad -> Target Encoding (con validación cruzada interna)
HIGH_CARD_FEATURES = ["Brand", "Model"]

FEATURES = NUMERIC_FEATURES + LOW_CARD_FEATURES + HIGH_CARD_FEATURES

# Valores que el dataset usa para indicar "dato no disponible".
MISSING_TOKENS = ["-", "- / -", "", "POA"]

# Precios por debajo de este umbral se consideran errores de digitación
# (regla fija definida a priori, no aprendida de los datos).
MIN_VALID_PRICE = 500


def _to_nan(s: pd.Series) -> pd.Series:
    """Quita espacios y convierte los marcadores de dato faltante en NaN."""
    s = s.astype(object)
    s = s.where(s.isna(), s.astype(str).str.strip())
    return s.where(~s.isin(MISSING_TOKENS), np.nan)


def _extract_number(s: pd.Series, pattern: str) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.extract(pattern, expand=False), errors="coerce")


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte registros crudos (como vienen en el CSV) en las variables del modelo.

    Acepta columnas faltantes (se tratan como nulos) para ser tolerante a
    entradas incompletas en fases posteriores (API).
    """
    df = df.copy()
    for col in RAW_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    out = pd.DataFrame(index=df.index)

    # --- Numéricas ---
    out["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    out["Kilometres"] = pd.to_numeric(_to_nan(df["Kilometres"]), errors="coerce")
    # "4 cyl, 2.2 L" -> 2.2  |  "0 L" (eléctricos) -> 0
    out["EngineLitres"] = _extract_number(df["Engine"], r"([\d.]+)\s*L")
    # "4 cyl" -> 4  |  "0 L" (eléctricos) -> 0
    cyl = _extract_number(df["CylindersinEngine"], r"(\d+)\s*cyl")
    is_zero_l = _to_nan(df["CylindersinEngine"]).eq("0 L")
    out["Cylinders"] = cyl.mask(is_zero_l, 0)
    # "8.7 L / 100 km" -> 8.7
    out["FuelConsumption"] = _extract_number(df["FuelConsumption"], r"([\d.]+)\s*L")
    # En algunos registros la columna Doors contiene asientos ("5 Seats"): se descartan.
    out["Doors"] = _extract_number(df["Doors"], r"(\d+)\s*Doors")
    out["Seats"] = _extract_number(df["Seats"], r"(\d+)\s*Seats")

    # --- Categóricas ---
    for col in ["UsedOrNew", "Transmission", "DriveType", "FuelType", "BodyType", "Brand", "Model"]:
        out[col] = _to_nan(df[col])
    # "Caringbah, NSW" -> "NSW" (la ciudad tiene demasiada cardinalidad)
    state = _to_nan(df["Location"]).str.split(",").str[-1]
    out["State"] = _to_nan(state.str.replace("AU-", "", regex=False))
    # "White / Black" -> "White" (color exterior)
    colour = _to_nan(_to_nan(df["ColourExtInt"]).str.split("/").str[0])
    # Algunos registros traen texto que no es un color (p. ej. "5 years").
    out["ColourExt"] = colour.where(~colour.str.contains(r"\d", na=False), np.nan)

    # Tipos float estándar (NaN) en lugar de los tipos "nullable" de pandas (pd.NA).
    out[NUMERIC_FEATURES] = out[NUMERIC_FEATURES].astype(float)
    return out[FEATURES]


def load_dataset(path) -> tuple[pd.DataFrame, pd.Series]:
    """Lee el CSV y devuelve (X crudo, y).

    Solo elimina filas cuyo *objetivo* no es utilizable (nulo, "POA" o precio
    inválido); las variables predictoras se dejan crudas para el Pipeline.
    """
    df = pd.read_csv(path)
    y = pd.to_numeric(df[TARGET], errors="coerce")
    valid = y.notna() & (y >= MIN_VALID_PRICE)
    return df.loc[valid, RAW_COLUMNS].reset_index(drop=True), y[valid].reset_index(drop=True)
