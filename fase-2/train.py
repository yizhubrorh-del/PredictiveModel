"""Entrenamiento reproducible y versionado del modelo de precios de vehículos."""
import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split

from src.features import FEATURES, MIN_VALID_PRICE, RAW_COLUMNS, TARGET
from src.pipeline import BEST_PARAMS, make_model

# Title y Car/Suv fueron excluidas del modelo en la Fase 1 por redundancia.
REQUIRED_COLUMNS = [c for c in RAW_COLUMNS if c not in {"Title", "Car/Suv"}]


def regression_metrics(y_true, y_pred) -> dict:
    """Métricas de regresión expresadas en AUD, salvo R² y MAPE."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MedAE": float(median_absolute_error(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def train(args: argparse.Namespace) -> Path:
    data_path = Path(args.data)
    if not data_path.is_file():
        raise ValueError(f"No existe el archivo CSV: {data_path}")
    if args.target != TARGET:
        raise ValueError(f"La variable objetivo de este proyecto debe ser '{TARGET}'.")
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size debe estar entre 0 y 1 (excluidos).")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.version) or args.version in {".", ".."}:
        raise ValueError("--version debe contener solo letras, números, guiones, puntos o '_'.")

    version_dir = Path(args.output_dir) / args.version
    if version_dir.exists():
        raise FileExistsError(f"La versión '{args.version}' ya existe: {version_dir}. Usa otra versión.")

    raw = pd.read_csv(data_path)
    missing = [c for c in [args.target, *REQUIRED_COLUMNS] if c not in raw.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en el CSV: {', '.join(missing)}")

    y_numeric = pd.to_numeric(raw[args.target], errors="coerce")
    valid = y_numeric.notna() & (y_numeric >= MIN_VALID_PRICE)
    X = raw.loc[valid].copy().reset_index(drop=True)
    y = y_numeric.loc[valid].reset_index(drop=True)
    # Usar siempre el orden del esquema original; columnas opcionales ausentes son nulas.
    X = X.reindex(columns=RAW_COLUMNS)
    if len(X) < 50:
        raise ValueError("Se requieren al menos 50 filas con precios válidos para entrenar.")

    # Igual que en la Fase 1: estratificar por deciles del precio cuando sea viable.
    price_bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
    counts = pd.Series(price_bins).value_counts()
    stratify = price_bins if len(counts) >= 2 and counts.min() >= 2 else None
    split_strategy = "deciles_precio" if stratify is not None else "aleatoria"
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size,
            random_state=args.seed, stratify=stratify,
        )
    except ValueError as exc:
        raise ValueError(f"No fue posible separar los datos: {exc}") from exc

    if len(X_train) < 10 or len(X_test) < 2:
        raise ValueError("La partición deja muy pocos datos para entrenamiento o prueba.")

    model = make_model(args.seed)
    print(f"Entrenando {len(X_train)} registros; prueba: {len(X_test)}...")
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    baseline_prediction = np.full(len(y_test), float(np.median(y_train)))
    metrics = {
        "model": regression_metrics(y_test, prediction),
        "baseline": regression_metrics(y_test, baseline_prediction),
        "mape_unit": "fraction (multiplicar por 100 para porcentaje)",
        "currency": "AUD",
    }

    metadata = {
        "version": args.version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "problem": "regression",
        "target": args.target,
        "algorithm": "HistGradientBoostingRegressor",
        "target_transform": "log1p / expm1",
        "hyperparameters": {**BEST_PARAMS, "random_state": args.seed},
        "input_columns": RAW_COLUMNS,
        "required_columns": REQUIRED_COLUMNS,
        "optional_columns": [c for c in RAW_COLUMNS if c not in REQUIRED_COLUMNS],
        "engineered_features": FEATURES,
        "data": {
            "source": "Kaggle - Australian Vehicle Prices",
            "file": str(data_path),
            "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "rows_raw": int(len(raw)),
            "rows_used": int(len(X)),
            "rows_train": int(len(X_train)),
            "rows_test": int(len(X_test)),
        },
        "split": {"strategy": split_strategy, "test_size": args.test_size, "seed": args.seed},
        "metric_primary": "MAE",
        "dependencies": "fase-2/requirements.txt",
        "python_version": platform.python_version(),
        "library_versions": {
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "git_commit": git_commit(),
        "training_note": "Hiperparámetros elegidos en Fase 1; sin nueva búsqueda en Fase 2.",
    }

    # Creamos la carpeta solamente después de entrenar correctamente.
    version_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, version_dir / "modelo.joblib", compress=3)
    (version_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (version_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    print(f"MAE modelo: {metrics['model']['MAE']:,.2f} AUD")
    print(f"MAE baseline: {metrics['baseline']['MAE']:,.2f} AUD")
    print(f"R²: {metrics['model']['R2']:.3f}")
    print(f"MAPE: {metrics['model']['MAPE']:.2%}")
    print(f"Versión guardada: {version_dir.resolve()}")
    return version_dir


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entrenar y versionar el modelo de precios australianos.")
    parser.add_argument("--data", required=True, help="Ruta al CSV etiquetado.")
    parser.add_argument("--target", default="Price", help="Variable objetivo (Price).")
    parser.add_argument("--output-dir", required=True, help="Carpeta de versiones del modelo.")
    parser.add_argument("--version", required=True, help="Identificador único, p. ej. v1.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporción de prueba.")
    return parser


if __name__ == "__main__":
    try:
        train(make_parser().parse_args())
    except (ValueError, FileExistsError, OSError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
