"""Pruebas básicas de Fase 2: entrenamiento, serialización y versionamiento.

Ejecutar desde fase-2 con: python -m pytest -v
Las pruebas usan datos sintéticos y carpetas temporales; no alteran models/.
"""

import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest

# Permitir que `python -m pytest` encuentre train y src desde fase-2.
FASE2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FASE2))

import train as training
from src.pipeline import make_model


def build_sample_dataset(n=100):
    """Dataset pequeño con el esquema real; sin descargas ni datos externos."""
    records = []
    for i in range(n):
        records.append({
            "Price": 8000 + (i * 420) + (i % 7) * 150,
            "Brand": ["Toyota", "Ford", "Honda", "Mazda"][i % 4],
            "Year": 2008 + i % 15,
            "Model": ["Corolla", "Focus", "Civic", "Mazda3"][i % 4],
            "UsedOrNew": "USED" if i % 5 else "NEW",
            "Transmission": "Automatic" if i % 2 else "Manual",
            "Engine": "4 cyl, 2.0 L",
            "DriveType": "Front",
            "FuelType": "Petrol",
            "FuelConsumption": "8.7 L / 100 km",
            "Kilometres": str(5000 + i * 1200),
            "ColourExtInt": "White / Black",
            "Location": "Sydney, NSW",
            "CylindersinEngine": "4 cyl",
            "BodyType": "Sedan",
            "Doors": "4 Doors",
            "Seats": "5 Seats",
        })
    return pd.DataFrame.from_records(records)


def fast_model(seed):
    """Mismo pipeline real, con menos iteraciones para que pytest sea rápido."""
    model = make_model(seed)
    model.regressor.set_params(model__max_iter=12)
    return model


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("trained_models")
    csv_path = root / "synthetic.csv"
    build_sample_dataset().to_csv(csv_path, index=False)
    models_dir = root / "models"

    current = {}

    def create_model(seed):
        model = fast_model(seed)
        current["model_before_serialization"] = model
        return model

    args = Namespace(
        data=str(csv_path), target="Price", output_dir=str(models_dir),
        version="v1", seed=42, test_size=0.2,
    )
    with patch.object(training, "make_model", side_effect=create_model):
        version_path = training.train(args)

    return {
        "root": root,
        "csv": csv_path,
        "models_dir": models_dir,
        "v1": version_path,
        "model_before_serialization": current["model_before_serialization"],
    }


def test_entrenamiento_y_almacenamiento(trained):
    version = trained["v1"]
    assert (version / "modelo.joblib").is_file()
    assert (version / "metrics.json").is_file()
    assert (version / "metadata.json").is_file()

    metrics = json.loads((version / "metrics.json").read_text(encoding="utf-8"))
    metadata = json.loads((version / "metadata.json").read_text(encoding="utf-8"))
    assert "model" in metrics and "baseline" in metrics
    assert metrics["model"]["MAE"] >= 0
    assert metrics["baseline"]["MAE"] >= 0
    assert metadata["version"] == "v1"
    assert metadata["target"] == "Price"


def test_carga_y_prediccion_equivalente(trained):
    sample = pd.read_csv(trained["csv"]).drop(columns=["Price"]).head(5)
    before = trained["model_before_serialization"].predict(sample)
    restored = joblib.load(trained["v1"] / "modelo.joblib")
    after = restored.predict(sample)
    assert len(after) == len(sample)
    np.testing.assert_allclose(after, before, rtol=1e-12, atol=1e-8)
    assert np.isfinite(after).all()


def test_valores_faltantes_admitidos(trained):
    sample = pd.read_csv(trained["csv"]).drop(columns=["Price"]).head(4)
    sample.loc[sample.index[0], "Engine"] = np.nan
    sample.loc[sample.index[1], "BodyType"] = np.nan
    model = joblib.load(trained["v1"] / "modelo.joblib")
    predictions = model.predict(sample)
    assert len(predictions) == len(sample)
    assert np.isfinite(predictions).all()


def test_entrada_invalida_columna_obligatoria(trained, tmp_path):
    incomplete = pd.read_csv(trained["csv"]).drop(columns=["Brand"])
    invalid_csv = tmp_path / "missing_brand.csv"
    incomplete.to_csv(invalid_csv, index=False)
    args = Namespace(
        data=str(invalid_csv), target="Price", output_dir=str(tmp_path / "models"),
        version="bad", seed=42, test_size=0.2,
    )
    with pytest.raises(ValueError, match="Faltan columnas obligatorias.*Brand"):
        training.train(args)
    assert not (tmp_path / "models" / "bad").exists()


def test_versionamiento_sin_sobrescritura(trained):
    v1_path = trained["v1"] / "modelo.joblib"
    original_sha = hashlib.sha256(v1_path.read_bytes()).hexdigest()
    args = Namespace(
        data=str(trained["csv"]), target="Price",
        output_dir=str(trained["models_dir"]), version="v2",
        seed=43, test_size=0.2,
    )
    with patch.object(training, "make_model", side_effect=fast_model):
        second_version = training.train(args)
    assert (second_version / "modelo.joblib").is_file()
    assert (trained["v1"] / "metadata.json").is_file()
    sample = pd.read_csv(trained["csv"]).drop(columns=["Price"]).head(3)
    for version in [trained["v1"], second_version]:
        predictions = joblib.load(version / "modelo.joblib").predict(sample)
        assert len(predictions) == 3
        assert np.isfinite(predictions).all()

    with pytest.raises(FileExistsError, match="ya existe"):
        training.train(args)
    assert hashlib.sha256(v1_path.read_bytes()).hexdigest() == original_sha

def test_source_commit_se_guarda_en_metadata(trained, tmp_path):
    """El SHA proporcionado debe registrarse en los metadatos."""
    sha = "A" * 40

    args = Namespace(
        data=str(trained["csv"]),
        target="Price",
        output_dir=str(tmp_path / "models"),
        version="v-traceability",
        seed=42,
        test_size=0.2,
        source_commit=sha,
    )

    with patch.object(training, "make_model", side_effect=fast_model):
        version_dir = training.train(args)

    metadata = json.loads(
        (version_dir / "metadata.json").read_text(encoding="utf-8")
    )

    assert metadata["git_commit"] == sha.lower()
    assert (version_dir / "modelo.joblib").is_file()


def test_source_commit_invalido_es_rechazado(trained, tmp_path):
    """Un SHA inválido no debe producir una versión del modelo."""
    output_dir = tmp_path / "models"

    args = Namespace(
        data=str(trained["csv"]),
        target="Price",
        output_dir=str(output_dir),
        version="v-invalid",
        seed=42,
        test_size=0.2,
        source_commit="commit-invalido",
    )

    with pytest.raises(ValueError, match="source-commit"):
        training.train(args)

    assert not (output_dir / "v-invalid").exists()
