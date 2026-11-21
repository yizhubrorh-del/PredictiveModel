"""Métricas de monitoreo: operación, predicciones, rendimiento y deriva de datos.

La deriva se mide con el PSI (Population Stability Index) entre la distribución
de referencia (dataset de entrenamiento) y la observada en producción:

    PSI = Σ (p_prod - p_ref) · ln(p_prod / p_ref)

Interpretación habitual: < 0,10 estable · 0,10-0,25 cambio moderado · > 0,25 cambio fuerte.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from monitoreo import politica
from monitoreo.store import MonitorStore

EPS = 1e-4
# Variable de producción -> (columna en predicciones, columna del dataset de referencia)
VARIABLES_DERIVA = {
    "Year": ("year", "Year"),
    "Kilometres": ("kilometres", "Kilometres"),
    "precio_predicho": ("prediccion", "Price"),
}


# ------------------------------ perfil de referencia ------------------------------
def construir_perfil(data_path: Path, n_bins: int = 10) -> dict:
    """Resume el dataset de entrenamiento: cortes por deciles, proporciones y marcas."""
    raw = pd.read_csv(data_path)
    price = pd.to_numeric(raw["Price"], errors="coerce")
    raw = raw.loc[price.notna() & (price >= 500)].assign(Price=price)
    numeric = {
        "Year": pd.to_numeric(raw["Year"], errors="coerce"),
        "Kilometres": pd.to_numeric(raw["Kilometres"], errors="coerce"),
        "Price": raw["Price"],
    }
    variables = {}
    for name, values in numeric.items():
        values = values.dropna().to_numpy(dtype=float)
        cortes = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1]))
        variables[name] = {
            "cortes": cortes.tolist(),
            "proporciones": _proporciones(values, cortes).tolist(),
            "media": float(values.mean()),
            "mediana": float(np.median(values)),
        }
    marcas = sorted(raw["Brand"].dropna().astype(str).str.strip().unique().tolist())
    return {
        "fuente": str(data_path.name),
        "filas": int(len(raw)),
        "creado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "variables": variables,
        "marcas": marcas,
    }


def cargar_perfil(path: Path) -> dict | None:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _proporciones(values: np.ndarray, cortes) -> np.ndarray:
    # Intervalos (-inf, c1], (c1, c2], ..., (ck, inf)
    idx = np.searchsorted(np.asarray(cortes, dtype=float), values, side="left")
    counts = np.bincount(idx, minlength=len(cortes) + 1)
    return counts / max(counts.sum(), 1)


def psi(p_ref, p_prod) -> float:
    p_ref = np.clip(np.asarray(p_ref, dtype=float), EPS, None)
    p_prod = np.clip(np.asarray(p_prod, dtype=float), EPS, None)
    return float(np.sum((p_prod - p_ref) * np.log(p_prod / p_ref)))


# ---------------------------------- resumen ----------------------------------
def _pct(values, q) -> float | None:
    return round(float(np.percentile(values, q)), 2) if len(values) else None


def info_versiones(models_dir: Path, activa: str) -> list[dict]:
    """Versiones guardadas por train.py con sus métricas de prueba (sin cargar modelos)."""
    versiones = []
    for d in sorted(Path(models_dir).iterdir()) if Path(models_dir).is_dir() else []:
        meta_p, metrics_p = d / "metadata.json", d / "metrics.json"
        if not (d / "modelo.joblib").is_file() or not meta_p.is_file():
            continue
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_p.read_text(encoding="utf-8")) if metrics_p.is_file() else {}
        model_m = metrics.get("model", {})
        versiones.append({
            "version": d.name,
            "activa": d.name == activa,
            "creado_utc": meta.get("created_at_utc"),
            "semilla": meta.get("split", {}).get("seed"),
            "git_commit": meta.get("git_commit"),
            "filas_entrenamiento": meta.get("data", {}).get("rows_train"),
            "MAE_prueba": model_m.get("MAE"),
            "R2_prueba": model_m.get("R2"),
            "MAPE_prueba": model_m.get("MAPE"),
        })
    return versiones


def resumen_operacion(req: pd.DataFrame) -> dict:
    total = len(req)
    pred = req[req["ruta"] == "/predict"] if total else req
    pred_ok = pred[pred["estado"] < 400] if total else pred
    return {
        "total": total,
        "por_ruta": req["ruta"].value_counts().to_dict() if total else {},
        "por_estado": {str(k): int(v) for k, v in req["estado"].value_counts().sort_index().items()} if total else {},
        "tasa_error_cliente": round(float((req["estado"].between(400, 499)).mean()), 4) if total else 0.0,
        "tasa_error_servidor": round(float((req["estado"] >= 500).mean()), 4) if total else 0.0,
        "latencia_predict_ms": {
            "p50": _pct(pred_ok["latencia_ms"], 50),
            "p95": _pct(pred_ok["latencia_ms"], 95),
            "max": _pct(pred_ok["latencia_ms"], 100),
        },
    }


def resumen_predicciones(pred: pd.DataFrame) -> dict:
    values = pred["prediccion"].to_numpy(dtype=float)
    return {
        "total": int(len(pred)),
        "por_version": pred["version"].value_counts().to_dict(),
        "precio_aud": {
            "media": round(float(values.mean()), 2) if len(values) else None,
            "p05": _pct(values, 5), "mediana": _pct(values, 50), "p95": _pct(values, 95),
        },
    }


def resumen_rendimiento(pred: pd.DataFrame, versiones: list[dict]) -> dict:
    """Error real en producción, solo para predicciones con precio real (POST /feedback)."""
    prueba = {v["version"]: v for v in versiones}
    etiquetadas = pred.dropna(subset=["precio_real"])
    por_version = {}
    for version, g in etiquetadas.groupby("version"):
        err = (g["prediccion"] - g["precio_real"]).abs()
        mae = float(err.mean())
        mape = float((err / g["precio_real"]).mean())
        ref_mae = prueba.get(version, {}).get("MAE_prueba")
        ref_mape = prueba.get(version, {}).get("MAPE_prueba")
        por_version[version] = {
            "n": int(len(g)),
            "MAE": round(mae, 2),
            "MAPE": round(mape, 4),
            "MAE_prueba": round(ref_mae, 2) if ref_mae else None,
            "MAPE_prueba": round(ref_mape, 4) if ref_mape else None,
            # > 1 significa que el modelo erra más en producción que en la prueba.
            "ratio_mae": round(mae / ref_mae, 3) if ref_mae else None,
            "ratio_mape": round(mape / ref_mape, 3) if ref_mape else None,
        }
    return {
        "predicciones_con_precio_real": int(len(etiquetadas)),
        "cobertura": round(len(etiquetadas) / len(pred), 4) if len(pred) else 0.0,
        "por_version": por_version,
    }


def resumen_deriva(pred: pd.DataFrame, perfil: dict | None) -> dict | None:
    if perfil is None:
        return None
    variables = {}
    for name, (col_prod, col_ref) in VARIABLES_DERIVA.items():
        ref = perfil["variables"][col_ref]
        values = pd.to_numeric(pred[col_prod], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) == 0:
            variables[name] = {"n": 0, "psi": None}
            continue
        p_prod = _proporciones(values, ref["cortes"])
        variables[name] = {
            "n": int(len(values)),
            "psi": round(psi(ref["proporciones"], p_prod), 4),
            "media_referencia": round(ref["media"], 2),
            "media_produccion": round(float(values.mean()), 2),
            "faltantes": round(1 - len(values) / len(pred), 4) if len(pred) else 0.0,
            "proporciones_referencia": ref["proporciones"],
            "proporciones_produccion": [round(float(x), 4) for x in p_prod],
            "cortes": ref["cortes"],
        }
    marcas = pred["brand"].dropna().astype(str).str.strip()
    conocidas = set(perfil["marcas"])
    nuevas = marcas[~marcas.isin(conocidas)]
    return {
        "variables": variables,
        "marcas_desconocidas": {
            "proporcion": round(float(len(nuevas) / len(marcas)), 4) if len(marcas) else 0.0,
            "ejemplos": sorted(nuevas.unique().tolist())[:10],
        },
    }


def calcular_resumen(store: MonitorStore, models_dir: Path, activa: str,
                     perfil: dict | None, desde: str | None = None) -> dict:
    req = store.requests(desde)
    pred = store.predictions(desde)
    versiones = info_versiones(models_dir, activa)
    resumen = {
        "generado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ventana": {
            "desde": desde,
            "primera_solicitud": req["ts"].min() if len(req) else None,
            "ultima_solicitud": req["ts"].max() if len(req) else None,
        },
        "modelo_activo": activa,
        "versiones": versiones,
        "operacion": resumen_operacion(req),
        "predicciones": resumen_predicciones(pred),
        "rendimiento": resumen_rendimiento(pred, versiones),
        "deriva": resumen_deriva(pred, perfil),
    }
    resumen["politica"] = politica.evaluar(resumen)
    return resumen
