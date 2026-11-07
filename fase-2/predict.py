"""Predicción de precios de vehículos con un modelo versionado por train.py."""
import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

PREDICTION_COLUMN = "prediccion"


class PredictionError(Exception):
    """Error de entrada esperado: se muestra al usuario sin traceback."""


def load_metadata(model_path: Path) -> dict:
    """Lee el metadata.json guardado junto al modelo por train.py."""
    metadata_path = model_path.parent / "metadata.json"
    if not metadata_path.is_file():
        raise PredictionError(
            f"No existe metadata.json junto al modelo: {metadata_path}. "
            "Usa una carpeta de versión generada por train.py."
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PredictionError(f"metadata.json no es un JSON válido: {exc}") from exc
    if not isinstance(metadata.get("required_columns"), list):
        raise PredictionError(f"metadata.json no define 'required_columns': {metadata_path}")
    return metadata


def load_model(model_path: Path):
    if not model_path.is_file():
        raise PredictionError(f"No existe el archivo del modelo: {model_path}")
    try:
        return joblib.load(model_path)
    except ModuleNotFoundError as exc:
        # El modelo referencia src.features.clean_raw, que debe ser importable.
        raise PredictionError(
            f"No se pudo cargar el modelo porque falta el módulo '{exc.name}'. "
            "Ejecuta predict.py desde su ubicación en fase-2/ y verifica las dependencias."
        ) from exc
    except Exception as exc:
        raise PredictionError(f"No se pudo cargar el modelo {model_path}: {exc}") from exc


def read_input(input_path: Path, metadata: dict) -> pd.DataFrame:
    if not input_path.is_file():
        raise PredictionError(f"No existe el archivo de entrada: {input_path}")
    try:
        data = pd.read_csv(input_path)
    except pd.errors.EmptyDataError as exc:
        raise PredictionError(f"El archivo de entrada está vacío: {input_path}") from exc
    except pd.errors.ParserError as exc:
        raise PredictionError(f"No se pudo leer el CSV de entrada: {exc}") from exc

    missing = [c for c in metadata["required_columns"] if c not in data.columns]
    if missing:
        raise PredictionError(f"Faltan columnas obligatorias en el CSV: {', '.join(missing)}")
    if data.empty:
        raise PredictionError(f"El archivo de entrada no contiene vehículos: {input_path}")

    # Mismo orden de columnas que en el entrenamiento; las opcionales ausentes
    # quedan como nulas y las columnas extra (p. ej. Price) se ignoran.
    return data.reindex(columns=metadata.get("input_columns", metadata["required_columns"]))


def predict(args: argparse.Namespace) -> Path:
    model_path = Path(args.model)
    input_path = Path(args.input)
    output_path = Path(args.output)

    model = load_model(model_path)
    metadata = load_metadata(model_path)
    X = read_input(input_path, metadata)

    predictions = model.predict(X)
    result = pd.DataFrame({PREDICTION_COLUMN: predictions})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"Modelo: {model_path} (versión {metadata.get('version', 'desconocida')})")
    print(f"Predicciones generadas: {len(result)}")
    print(f"Archivo guardado: {output_path.resolve()}")
    return output_path


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predecir el precio (AUD) de vehículos con un modelo entrenado por train.py.",
    )
    parser.add_argument(
        "--model", required=True,
        help="Ruta a modelo.joblib; metadata.json debe estar en la misma carpeta.",
    )
    parser.add_argument("--input", required=True, help="CSV con los vehículos a predecir (sin Price).")
    parser.add_argument("--output", required=True, help="CSV de salida con la columna 'prediccion'.")
    return parser


if __name__ == "__main__":
    try:
        predict(make_parser().parse_args())
    except PredictionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except OSError as exc:
        print(f"ERROR: no se pudo acceder al archivo: {exc}", file=sys.stderr)
        sys.exit(1)
