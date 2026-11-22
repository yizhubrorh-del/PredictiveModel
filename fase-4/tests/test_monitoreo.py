"""Pruebas de la Fase 4: registro, métricas, versiones, política y script de monitoreo.

Ejecutar desde fase-4 con:  python -m pytest -v
Usan datos sintéticos, modelos rápidos y carpetas temporales (no tocan fase-2/models).
"""
import importlib.util
import sys
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

FASE4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FASE4))

from monitoreo.config import FASE2_DIR, Settings  # noqa: E402  (agrega fase-3 y fase-2)
from monitoreo import metricas, politica  # noqa: E402
from monitoreo.store import MonitorStore  # noqa: E402
from api_monitoreo import create_app  # noqa: E402
import monitor  # noqa: E402
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
    "Brand": "Toyota", "Model": "Corolla", "Year": 2015, "UsedOrNew": "USED",
    "Transmission": "Automatic", "Engine": "4 cyl, 2.0 L", "DriveType": "Front",
    "FuelType": "Petrol", "FuelConsumption": "8.7 L / 100 km", "Kilometres": 60000,
    "ColourExtInt": "White / Black", "Location": "Sydney, NSW", "CylindersinEngine": "4 cyl",
    "BodyType": "Sedan", "Doors": "4 Doors", "Seats": "5 Seats",
}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Dos versiones rápidas (v1, v2) y un perfil de referencia del dataset sintético."""
    root = tmp_path_factory.mktemp("fase4")
    csv_path = root / "synthetic.csv"
    build_sample_dataset(200).to_csv(csv_path, index=False)
    models_dir = root / "models"
    with patch.object(training, "make_model", side_effect=fast_model):
        for version, seed in (("v1", 42), ("v2", 43)):
            training.train(Namespace(data=str(csv_path), target="Price", output_dir=str(models_dir),
                                     version=version, seed=seed, test_size=0.2))
    perfil_path = root / "perfil.json"
    perfil_path.write_text(__import__("json").dumps(metricas.construir_perfil(csv_path)), encoding="utf-8")
    return {"settings": Settings(models_dir=models_dir, data_path=csv_path, default_version="v1"),
            "perfil": perfil_path, "csv": csv_path}


@pytest.fixture()
def api(env, tmp_path):
    app = create_app(env["settings"], db_path=tmp_path / "monitoreo.db", perfil_path=env["perfil"])
    return TestClient(app, raise_server_exceptions=False)


# ------------------------- registro de predicciones -------------------------
def test_predict_registra_y_devuelve_ids(api):
    response = api.post("/predict", json={"vehicles": [VEHICLE, {**VEHICLE, "Year": 2020}]})
    assert response.status_code == 200
    ids = response.json()["ids_prediccion"]
    assert len(ids) == 2 and len(set(ids)) == 2
    logged = api.app.state.store.predictions()
    assert set(logged["id"]) == set(ids)
    assert logged["version"].eq("v1").all()
    assert sorted(logged["year"]) == [2015, 2020]
    assert logged["prediccion"].round(2).tolist() == response.json()["predicciones"]


def test_todas_las_solicitudes_se_registran(api):
    api.get("/health")
    api.post("/predict", json={"vehicles": []})                        # 422
    api.post("/predict?version=v99", json={"vehicles": [VEHICLE]})      # 404
    api.get("/docs")                                                    # no se registra
    req = api.app.state.store.requests()
    assert req[["ruta", "estado"]].values.tolist() == [["/health", 200], ["/predict", 422], ["/predict", 404]]
    assert (req["latencia_ms"] >= 0).all()
    assert len(api.app.state.store.predictions()) == 0  # las fallidas no generan predicciones


def test_fallo_del_registro_no_impide_predecir(api):
    with patch.object(api.app.state.store, "log_predictions", side_effect=OSError("disco lleno")):
        response = api.post("/predict", json={"vehicles": [VEHICLE]})
    assert response.status_code == 200
    assert "ids_prediccion" not in response.json()


# --------------------------------- feedback ---------------------------------
def test_feedback(api):
    pid = api.post("/predict", json={"vehicles": [VEHICLE]}).json()["ids_prediccion"][0]
    response = api.post("/feedback", json={"id_prediccion": pid, "precio_real": 15000})
    assert response.status_code == 201
    assert response.json()["error_absoluto"] >= 0
    assert api.post("/feedback", json={"id_prediccion": pid, "precio_real": 15000}).status_code == 409
    assert api.post("/feedback", json={"id_prediccion": "0" * 32, "precio_real": 1}).status_code == 404


@pytest.mark.parametrize("payload", [
    {"id_prediccion": "no-es-un-id", "precio_real": 1000},
    {"id_prediccion": "a" * 32, "precio_real": -5},
    {"id_prediccion": "a" * 32},
])
def test_feedback_invalido(api, payload):
    assert api.post("/feedback", json=payload).status_code == 422


# --------------------------------- métricas ---------------------------------
def test_metrics(api):
    ids = api.post("/predict", json={"vehicles": [VEHICLE] * 4}).json()["ids_prediccion"]
    for pid in ids[:2]:
        api.post("/feedback", json={"id_prediccion": pid, "precio_real": 20000})
    api.post("/predict", json={"vehicles": []})
    body = api.get("/metrics").json()
    assert body["modelo_activo"] == "v1"
    assert body["predicciones"]["total"] == 4
    assert body["operacion"]["por_estado"]["422"] == 1
    assert body["rendimiento"]["predicciones_con_precio_real"] == 2
    assert body["rendimiento"]["por_version"]["v1"]["ratio_mae"] > 0
    assert set(body["deriva"]["variables"]) == {"Year", "Kilometres", "precio_predicho"}
    assert body["politica"]["decision"] in politica.NIVELES


def test_metrics_ventana(api):
    api.post("/predict", json={"vehicles": [VEHICLE]})
    assert api.get("/metrics?horas=1").json()["predicciones"]["total"] == 1
    assert api.get("/metrics?horas=0").status_code == 422


# --------------------------------- versiones ---------------------------------
def test_models_y_cambio_de_version(api, env, tmp_path):
    body = api.get("/models").json()
    assert body["activa"] == "v1"
    assert [v["version"] for v in body["versiones"]] == ["v1", "v2"]

    response = api.put("/models/active", json={"version": "v2"})
    assert response.json() == {"activa": "v2", "anterior": "v1"}
    assert api.post("/predict", json={"vehicles": [VEHICLE]}).json()["version"] == "v2"
    assert api.get("/health").json()["modelo_por_defecto"] == "v2"
    assert api.get("/models").json()["historial"][0]["version"] == "v2"

    # La versión activa persiste al reiniciar la API con la misma base.
    restarted = TestClient(create_app(env["settings"], db_path=tmp_path / "monitoreo.db",
                                      perfil_path=env["perfil"]))
    assert restarted.post("/predict", json={"vehicles": [VEHICLE]}).json()["version"] == "v2"

    # Reversión
    assert api.put("/models/active", json={"version": "v1"}).json()["anterior"] == "v2"


def test_activar_version_inexistente_o_invalida(api):
    assert api.put("/models/active", json={"version": "v99"}).status_code == 404
    assert api.put("/models/active", json={"version": "../x"}).status_code == 422
    assert api.get("/models").json()["activa"] == "v1"


# ----------------------------------- PSI -----------------------------------
def test_psi():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    cortes = np.quantile(ref, np.linspace(0, 1, 11)[1:-1])
    p_ref = metricas._proporciones(ref, cortes)
    assert metricas.psi(p_ref, p_ref) == pytest.approx(0)
    assert metricas.psi(p_ref, metricas._proporciones(rng.normal(0, 1, 5000), cortes)) < 0.02
    assert metricas.psi(p_ref, metricas._proporciones(rng.normal(1.5, 1, 5000), cortes)) > 0.25


# --------------------------------- política ---------------------------------
def _resumen(n_pred=1000, psi=0.01, n_real=500, ratio_mae=1.0, ratio_mape=1.0,
             dias=10, err5xx=0.0, p95=50.0, marcas=0.0):
    creado = (datetime.now(timezone.utc) - timedelta(days=dias)).isoformat()
    var = {"psi": psi}
    return {
        "modelo_activo": "v1",
        "versiones": [{"version": "v1", "creado_utc": creado}],
        "predicciones": {"total": n_pred},
        "deriva": {"variables": {"Year": var, "Kilometres": {"psi": 0.01}},
                   "marcas_desconocidas": {"proporcion": marcas}},
        "rendimiento": {"por_version": {"v1": {
            "n": n_real, "MAE": 5000 * ratio_mae, "MAE_prueba": 5000, "ratio_mae": ratio_mae,
            "MAPE": 0.13 * ratio_mape, "MAPE_prueba": 0.13, "ratio_mape": ratio_mape}}},
        "operacion": {"tasa_error_servidor": err5xx, "latencia_predict_ms": {"p95": p95}},
    }


@pytest.mark.parametrize("kwargs, esperado, regla", [
    ({}, "OK", None),
    ({"psi": 0.15}, "VIGILAR", "R1-deriva"),
    ({"psi": 0.40}, "REENTRENAR", "R1-deriva"),
    ({"psi": 0.40, "n_pred": 100}, "OK", None),           # muestra insuficiente
    ({"marcas": 0.10}, "VIGILAR", "R1-deriva"),
    ({"ratio_mae": 1.25}, "VIGILAR", "R2-rendimiento"),
    ({"ratio_mape": 1.50}, "REENTRENAR", "R2-rendimiento"),
    ({"ratio_mae": 2.0, "n_real": 50}, "OK", None),       # muestra insuficiente
    ({"dias": 200}, "REENTRENAR", "R3-antiguedad"),
    ({"err5xx": 0.05}, "VIGILAR", "R4-operacion"),
    ({"p95": 5000}, "VIGILAR", "R4-operacion"),
])
def test_politica(kwargs, esperado, regla):
    result = politica.evaluar(_resumen(**kwargs))
    assert result["decision"] == esperado
    if regla:
        assert any(m["regla"] == regla and m["nivel"] == esperado for m in result["motivos"])
    assert result["motivos"][0]["nivel"] == esperado  # ordenados del más grave al menos grave


def test_politica_sin_perfil():
    resumen = _resumen()
    resumen["deriva"] = None
    assert politica.evaluar(resumen)["decision"] == "VIGILAR"


# ------------------------------ script de monitoreo ------------------------------
def test_reporte_genera_archivos(api, env, tmp_path):
    ids = api.post("/predict", json={"vehicles": [VEHICLE] * 3}).json()["ids_prediccion"]
    api.post("/feedback", json={"id_prediccion": ids[0], "precio_real": 18000})
    salida = tmp_path / "reporte"
    args = Namespace(db=str(api.app.state.store.path), perfil=str(env["perfil"]),
                     models_dir=str(env["settings"].models_dir), horas=None, salida=str(salida))
    assert monitor.cmd_reporte(args) == 0
    texto = (salida / "reporte_monitoreo.md").read_text(encoding="utf-8")
    assert "Decisión de la política" in texto and "`v1`" in texto
    assert (salida / "resumen.json").is_file()
    assert (salida / "deriva.png").is_file()


def test_reporte_codigo_2_si_reentrenar(api, env, tmp_path):
    with patch.object(politica, "evaluar", return_value={"decision": "REENTRENAR", "motivos": [], "umbrales": {}}):
        args = Namespace(db=str(api.app.state.store.path), perfil=str(env["perfil"]),
                         models_dir=str(env["settings"].models_dir), horas=None, salida=str(tmp_path))
        assert monitor.cmd_reporte(args) == 2


def test_reporte_sin_base(tmp_path):
    args = Namespace(db=str(tmp_path / "no.db"), perfil="x", models_dir=None, horas=None, salida=str(tmp_path))
    assert monitor.cmd_reporte(args) == 1


def test_comparar_versiones(env):
    import pandas as pd
    tabla = monitor.evaluar_versiones(pd.read_csv(env["csv"]), ["v1", "v2"], env["settings"].models_dir)
    assert set(tabla["version"]) == {"v1", "v2"}
    assert tabla["MAE"].is_monotonic_increasing
    with pytest.raises(ValueError, match="v99"):
        monitor.evaluar_versiones(pd.read_csv(env["csv"]), ["v99"], env["settings"].models_dir)


def test_store_persistencia(tmp_path):
    store = MonitorStore(tmp_path / "m.db")
    ids = store.log_predictions("v1", [{"Brand": "Kia", "Year": 2019, "Kilometres": 10.0}], [12345.0])
    store.log_event("activacion", "v2", {"anterior": "v1"})
    again = MonitorStore(tmp_path / "m.db")
    assert again.predictions()["id"].tolist() == ids
    assert again.active_version() == "v2"


def test_exportar_precios_reales(api, tmp_path):
    ids = api.post("/predict", json={"vehicles": [VEHICLE, {**VEHICLE, "Brand": "Ford"}]}).json()["ids_prediccion"]
    api.post("/feedback", json={"id_prediccion": ids[1], "precio_real": 21000})
    data = monitor.exportar_etiquetadas(api.app.state.store)
    assert len(data) == 1
    assert data.loc[0, "Brand"] == "Ford" and data.loc[0, "Price"] == 21000
    # El CSV exportado sirve directamente para comparar versiones (mismo esquema).
    out = tmp_path / "prod.csv"
    args = Namespace(db=str(api.app.state.store.path), horas=None, salida=str(out))
    assert monitor.cmd_exportar(args) == 0 and out.is_file()
