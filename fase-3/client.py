"""Cliente de prueba de la API REST (Fase 3).

Ejemplos (con la API en ejecución):
    python client.py health
    python client.py predict
    python client.py predict --input ../fase-2/data/ejemplo_prediccion.csv --version v2
    python client.py train --version v-api-001 --seed 42
    python client.py demo          # health + predict + una petición inválida
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import requests

DEFAULT_URL = "http://localhost:8000"
DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "fase-2" / "data" / "ejemplo_prediccion.csv"

# Columnas que acepta /predict (las del CSV original, sin Price).
API_FIELDS = [
    "Brand", "Year", "Model", "Car/Suv", "Title", "UsedOrNew", "Transmission",
    "Engine", "DriveType", "FuelType", "FuelConsumption", "Kilometres",
    "ColourExtInt", "Location", "CylindersinEngine", "BodyType", "Doors", "Seats",
]
MISSING = {"", "-", "- / -", "POA"}


def _number(value: str):
    try:
        return float(value)
    except ValueError:
        return None


def csv_to_payload(path: Path) -> dict:
    """Convierte las filas del CSV en el JSON de /predict (vacíos -> null)."""
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    vehicles = []
    for row in rows:
        vehicle = {}
        for field in API_FIELDS:
            value = (row.get(field) or "").strip()
            if value in MISSING:
                continue
            if field == "Year":
                vehicle[field] = int(float(value))
            elif field == "Kilometres":
                km = _number(value)
                if km is not None:
                    vehicle[field] = km
            else:
                vehicle[field] = value
        vehicles.append(vehicle)
    return {"vehicles": vehicles}


def call(method: str, url: str, **kwargs) -> int:
    """Hace la petición, imprime estado y cuerpo; devuelve el código HTTP."""
    try:
        response = requests.request(method, url, **kwargs)
    except requests.ConnectionError:
        print(f"ERROR: no se pudo conectar con {url}. ¿Está la API en ejecución?", file=sys.stderr)
        sys.exit(2)
    print(f"{method} {url} -> {response.status_code}")
    try:
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    except ValueError:
        print(response.text)
    return response.status_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Cliente de prueba de la API de precios.")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"URL base (por defecto {DEFAULT_URL}).")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("health", help="GET /health")

    p_predict = sub.add_parser("predict", help="POST /predict con un CSV")
    p_predict.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="CSV de vehículos.")
    p_predict.add_argument("--version", default=None, help="Versión del modelo (opcional).")

    p_train = sub.add_parser("train", help="POST /train")
    p_train.add_argument("--version", required=True, help="Identificador nuevo, p. ej. v-api-001.")
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--test-size", type=float, default=0.2)
    p_train.add_argument("--source-commit", default=None, help="SHA de 40 caracteres (opcional).")

    sub.add_parser("demo", help="Recorrido: health, predict y una entrada inválida")

    args = parser.parse_args()
    base = args.url.rstrip("/")

    if args.command == "health":
        status = call("GET", f"{base}/health", timeout=30)
    elif args.command == "predict":
        params = {"version": args.version} if args.version else None
        status = call("POST", f"{base}/predict", json=csv_to_payload(args.input),
                      params=params, timeout=60)
    elif args.command == "train":
        body = {"version": args.version, "seed": args.seed, "test_size": args.test_size}
        if args.source_commit:
            body["source_commit"] = args.source_commit
        print("Entrenando (puede tardar varios minutos)...")
        status = call("POST", f"{base}/train", json=body, timeout=1800)
    else:  # demo
        print("1) Estado del servicio")
        call("GET", f"{base}/health", timeout=30)
        print("\n2) Predicción con el CSV de ejemplo")
        status = call("POST", f"{base}/predict", json=csv_to_payload(DEFAULT_INPUT), timeout=60)
        print("\n3) Entrada inválida (falta Brand, año futuro): se espera 422")
        call("POST", f"{base}/predict",
             json={"vehicles": [{"Model": "Corolla", "Year": 2999}]}, timeout=30)
        print("\n4) Versión inexistente: se espera 404")
        call("POST", f"{base}/predict", params={"version": "no-existe"},
             json=csv_to_payload(DEFAULT_INPUT), timeout=30)

    return 0 if status < 400 else 1


if __name__ == "__main__":
    sys.exit(main())
