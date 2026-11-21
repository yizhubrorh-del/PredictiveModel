"""Simula tráfico de producción contra la API con monitoreo, para generar el reporte.

Envía vehículos reales de la partición de PRUEBA de v1 (no vistos en su entrenamiento),
registra su precio real mediante /feedback e incluye algunas solicitudes inválidas.

Escenarios:
  normal  muestra aleatoria de la partición de prueba (misma distribución que el entrenamiento)
  deriva  solo vehículos de 2012 o anteriores (cambio en la población de entrada)

Ejemplos (desde fase-4):
    python simular_trafico.py --escenario normal --db logs/normal.db
    python simular_trafico.py --escenario deriva --db logs/deriva.db
    python simular_trafico.py --escenario normal --url http://localhost:8000   # API en ejecución
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from monitoreo.config import REPO_ROOT
import client  # fase-3/client.py: conversión de CSV al JSON de /predict
from src.features import MIN_VALID_PRICE, RAW_COLUMNS  # fase-2

DATASET = REPO_ROOT / "data" / "raw" / "australian_vehicle_prices.csv"


def particion_prueba(data_path: Path, seed: int = 42, test_size: float = 0.2) -> pd.DataFrame:
    """Reproduce la partición de prueba de fase-2/train.py (deciles de precio, misma semilla)."""
    raw = pd.read_csv(data_path)
    y = pd.to_numeric(raw["Price"], errors="coerce")
    valid = y.notna() & (y >= MIN_VALID_PRICE)
    X = raw.loc[valid].reset_index(drop=True).reindex(columns=RAW_COLUMNS)
    y = y.loc[valid].reset_index(drop=True)
    bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
    _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=bins)
    return X_test.assign(Price=y_test.to_numpy())


def seleccionar(test: pd.DataFrame, escenario: str, n: int, seed: int) -> pd.DataFrame:
    if escenario == "deriva":
        test = test[pd.to_numeric(test["Year"], errors="coerce") <= 2012]
    return test.sample(n=min(n, len(test)), random_state=seed).reset_index(drop=True)


class Cliente:
    """Mismo uso con la API en proceso (TestClient) o una API remota (requests)."""

    def __init__(self, url: str | None, db: Path):
        if url:
            import requests
            self.session, self.base = requests.Session(), url.rstrip("/")
            self.extra = {"timeout": 120}
        else:
            from fastapi.testclient import TestClient
            from api_monitoreo import create_app
            self.session, self.base = TestClient(create_app(db_path=db)), ""
            self.extra = {}

    def post(self, path: str, **kwargs):
        return self.session.post(self.base + path, **self.extra, **kwargs)

    def get(self, path: str, **kwargs):
        return self.session.get(self.base + path, **self.extra, **kwargs)


def simular(args) -> int:
    rng = np.random.default_rng(args.seed)
    muestra = seleccionar(particion_prueba(Path(args.data)), args.escenario, args.n, args.seed)
    api = Cliente(args.url, Path(args.db))
    api.get("/health")

    enviados = etiquetados = 0
    with tempfile.TemporaryDirectory() as tmp:
        for start in range(0, len(muestra), args.lote):
            lote = muestra.iloc[start:start + args.lote]
            csv_path = Path(tmp) / "lote.csv"
            lote.to_csv(csv_path, index=False)
            response = api.post("/predict", json=client.csv_to_payload(csv_path))
            response.raise_for_status()
            ids = response.json().get("ids_prediccion") or []
            enviados += len(ids)
            # Solo una parte de los precios reales llega de vuelta (como en producción).
            for pid, precio in zip(ids, lote["Price"]):
                if rng.random() < args.cobertura:
                    api.post("/feedback", json={"id_prediccion": pid, "precio_real": float(precio)})
                    etiquetados += 1

    # Algunas solicitudes inválidas para poblar las métricas de errores.
    invalidas = [
        {"vehicles": []},
        {"vehicles": [{"Model": "Corolla", "Year": 2018}]},
        {"vehicles": [{"Brand": "Toyota", "Model": "Corolla", "Year": 2999}]},
        {"vehicles": [{"Brand": "Toyota", "Model": "Corolla", "Year": 2018, "Kilometres": -1}]},
    ]
    for payload in invalidas:
        api.post("/predict", json=payload)
    api.post("/predict", params={"version": "no-existe"}, json={"vehicles": [{"Brand": "Kia", "Model": "Rio", "Year": 2019}]})
    api.get("/metrics")

    print(f"Escenario '{args.escenario}': {enviados} predicciones, {etiquetados} precios reales, "
          f"{len(invalidas) + 1} solicitudes inválidas.")
    print(f"Base de monitoreo: {Path(args.db).resolve() if not args.url else args.url}")
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simular tráfico de producción para el monitoreo.")
    parser.add_argument("--escenario", choices=["normal", "deriva"], default="normal")
    parser.add_argument("--n", type=int, default=800, help="Vehículos a enviar (máximo disponible si hay menos).")
    parser.add_argument("--lote", type=int, default=25, help="Vehículos por solicitud.")
    parser.add_argument("--cobertura", type=float, default=0.7, help="Fracción con precio real.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data", default=str(DATASET))
    parser.add_argument("--db", default="logs/monitoreo.db", help="Base SQLite (modo en proceso).")
    parser.add_argument("--url", default=None, help="URL de una API en ejecución (opcional).")
    return parser


if __name__ == "__main__":
    sys.exit(simular(make_parser().parse_args()))
