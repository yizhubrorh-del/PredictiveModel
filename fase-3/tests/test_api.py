"""Pruebas de la API REST (Fase 3).

Ejecutar desde fase-3 con:  python -m pytest -v
Usan datos sintéticos y carpetas temporales (no modifican fase-2/models).
Reutilizan el generador de datos y el modelo rápido de las pruebas de la Fase 2.
"""
import importlib.util
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

FASE3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FASE3))

from app.config import FASE2_DIR, Settings  # noqa: E402  (agrega fase-2 al sys.path)
from app.main import create_app  # noqa: E402
import client  # noqa: E402
import train as training  # noqa: E402  (fase-2/train.py)


def _load_fase2_tests():
    spec = importlib.util.spec_from_file_location(
        "fase2_test_aplicacion", FASE2_DIR / "tests" / "test_aplicacion.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fase2_tests = _load_fase2_tests()
build_sample_dataset = fase2_tests.build_sample_dataset
fast_model = fase2_tests.fast_model

VEHICLE = {
    "Brand": "Toyota", "Model": "Corolla", "Year": 2015,
    "UsedOrNew": "USED", "Transmission": "Automatic", "Engine": "4 cyl, 2.0 L",
    "DriveType": "Front", "FuelType": "Petrol", "FuelConsumption": "8.7 L / 100 km",
    "Kilometres": 60000, "ColourExtInt": "White / Black", "Location": "Sydney, NSW",
    "CylindersinEngine": "4 cyl", "BodyType": "Sedan", "Doors": "4 Doors", "Seats": "5 Seats",
}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Directorio de modelos temporal con una versión 'v1' entrenada rápidamente."""
    root = tmp_path_factory.mktemp("api")
    csv_path = root / "synthetic.csv"
    build_sample_dataset().to_csv(csv_path, index=False)
    models_dir = root / "models"
    args = Namespace(data=str(csv_path), target="Price", output_dir=str(models_dir),
                     version="v1", seed=42, test_size=0.2)
    with patch.object(training, "make_model", side_effect=fast_model):
        training.train(args)
    return Settings(models_dir=models_dir, data_path=csv_path, default_version="v1")


@pytest.fixture()
def api(env):
    return TestClient(create_app(env), raise_server_exceptions=False)


# ------------------------------- /health -------------------------------
def test_health_ok(api):
    response = api.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["modelo_cargado"] is True
    assert "v1" in body["versiones_disponibles"]
    assert body["dataset_disponible"] is True


def test_health_degradado_sin_modelo(tmp_path):
    settings = Settings(models_dir=tmp_path, data_path=tmp_path / "no.csv", default_version="v1")
    response = TestClient(create_app(settings)).get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "degradado"


# ------------------------------- /predict -------------------------------
def test_predict_varios_vehiculos(api):
    second = {**VEHICLE, "Brand": "Ford", "Model": "Focus", "Year": 2020}
    response = api.post("/predict", json={"vehicles": [VEHICLE, second]})
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == "v1" and body["moneda"] == "AUD" and body["n"] == 2
    assert all(p > 0 for p in body["predicciones"])


def test_predict_solo_campos_obligatorios_y_numeros(api):
    """Opcionales ausentes se imputan; Doors/Seats admiten números."""
    vehicle = {"Brand": "Mazda", "Model": "Mazda3", "Year": 2019, "Doors": 4, "Seats": 5}
    response = api.post("/predict", json={"vehicles": [vehicle]})
    assert response.status_code == 200
    assert response.json()["n"] == 1


def test_predict_formato_numerico_equivale_al_texto(api):
    as_text = api.post("/predict", json={"vehicles": [VEHICLE]}).json()["predicciones"]
    as_numbers = {**VEHICLE, "Doors": 4, "Seats": 5, "CylindersinEngine": 4, "FuelConsumption": 8.7}
    assert api.post("/predict", json={"vehicles": [as_numbers]}).json()["predicciones"] == as_text


@pytest.mark.parametrize("payload, campo", [
    ({"vehicles": [{k: v for k, v in VEHICLE.items() if k != "Brand"}]}, "vehicles.0.Brand"),
    ({"vehicles": [{**VEHICLE, "Year": 2999}]}, "vehicles.0.Year"),
    ({"vehicles": [{**VEHICLE, "Year": "dos mil"}]}, "vehicles.0.Year"),
    ({"vehicles": [{**VEHICLE, "Kilometres": -5}]}, "vehicles.0.Kilometres"),
    ({"vehicles": [{**VEHICLE, "Brand": "   "}]}, "vehicles.0.Brand"),
    ({"vehicles": [{**VEHICLE, "Price": 20000}]}, "vehicles.0.Price"),
    ({"vehicles": []}, "vehicles"),
    ({}, "vehicles"),
])
def test_predict_entradas_invalidas(api, payload, campo):
    response = api.post("/predict", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "Entrada inválida"
    assert campo in [d["campo"] for d in body["detalles"]]


def test_predict_json_mal_formado(api):
    response = api.post("/predict", content="{no es json",
                        headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_predict_version_inexistente(api):
    response = api.post("/predict?version=v99", json={"vehicles": [VEHICLE]})
    assert response.status_code == 404
    assert "v99" in response.json()["error"]


def test_predict_version_invalida(api):
    response = api.post("/predict?version=../secreto", json={"vehicles": [VEHICLE]})
    assert response.status_code == 422


def test_error_interno_no_expone_detalles(api, env):
    loaded = api.app.state.registry.get("v1")
    with patch.object(loaded.model, "predict", side_effect=RuntimeError("fallo secreto")):
        response = api.post("/predict", json={"vehicles": [VEHICLE]})
    assert response.status_code == 500
    assert response.json() == {"error": "Error interno del servidor"}


def test_ruta_inexistente(api):
    response = api.get("/no-existe")
    assert response.status_code == 404
    assert response.json() == {"error": "Recurso no encontrado"}


# ------------------------------- /train -------------------------------
def test_train_crea_version_y_se_puede_usar(api, env):
    with patch.object(training, "make_model", side_effect=fast_model):
        response = api.post("/train", json={"version": "v-api", "seed": 7, "source_commit": "B" * 40})
    assert response.status_code == 201
    body = response.json()
    assert body["version"] == "v-api"
    assert body["metricas"]["modelo"]["MAE"] >= 0
    assert body["git_commit"] == "b" * 40
    assert (env.models_dir / "v-api" / "modelo.joblib").is_file()

    response = api.post("/predict?version=v-api", json={"vehicles": [VEHICLE]})
    assert response.status_code == 200 and response.json()["version"] == "v-api"


def test_train_version_duplicada(api, env):
    model_file = env.models_dir / "v1" / "modelo.joblib"
    before = model_file.read_bytes()
    response = api.post("/train", json={"version": "v1"})
    assert response.status_code == 409
    assert model_file.read_bytes() == before


@pytest.mark.parametrize("payload", [
    {"version": "../fuera"},
    {"version": "v-x", "test_size": 1.5},
    {"version": "v-x", "seed": -1},
    {"version": "v-x", "source_commit": "abc"},
    {"version": "v-x", "data": "/etc/passwd"},
    {},
])
def test_train_parametros_invalidos(api, env, payload):
    response = api.post("/train", json=payload)
    assert response.status_code == 422
    assert not (env.models_dir / "v-x").exists()


def test_train_sin_dataset(env, tmp_path):
    settings = Settings(models_dir=env.models_dir, data_path=tmp_path / "no.csv", default_version="v1")
    response = TestClient(create_app(settings)).post("/train", json={"version": "v-sin-datos"})
    assert response.status_code == 503


def test_train_en_curso(api):
    lock = api.app.state.train_lock
    lock.acquire()
    try:
        response = api.post("/train", json={"version": "v-ocupado"})
    finally:
        lock.release()
    assert response.status_code == 409
    assert "en curso" in response.json()["error"]


# --------------------- Modelo oficial y cliente de prueba ---------------------
def test_modelo_oficial_v1_reproduce_fase2():
    """La API devuelve lo mismo que predict.py de la Fase 2 con el modelo v1 versionado."""
    official = Settings(models_dir=FASE2_DIR / "models", data_path=Path("no-usado"),
                        default_version="v1")
    payload = client.csv_to_payload(FASE2_DIR / "data" / "ejemplo_prediccion.csv")
    response = TestClient(create_app(official)).post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["predicciones"] == pytest.approx([61107.88, 20053.32, 79467.43], abs=0.01)


def test_cliente_convierte_csv(tmp_path):
    csv_path = tmp_path / "entrada.csv"
    sample = build_sample_dataset(3)
    sample.loc[0, "Engine"] = "-"
    sample.to_csv(csv_path, index=False)
    payload = client.csv_to_payload(csv_path)
    assert len(payload["vehicles"]) == 3
    assert "Price" not in payload["vehicles"][0]
    assert "Engine" not in payload["vehicles"][0]
    assert isinstance(payload["vehicles"][0]["Year"], int)
