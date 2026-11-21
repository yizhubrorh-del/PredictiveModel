"""Script de monitoreo del modelo (Fase 4).

Comandos (desde fase-4):
    python monitor.py perfil                       # perfil de referencia desde el dataset
    python monitor.py reporte --salida reportes/   # reporte Markdown + JSON + figura
    python monitor.py comparar --data <csv> --versiones v1 v2
    python monitor.py exportar --salida datos_produccion.csv   # precios reales -> CSV

`reporte` lee la base SQLite donde la API registra solicitudes y predicciones,
calcula las métricas y aplica la política de reentrenamiento. Código de salida:
0 si la decisión es OK o VIGILAR, 2 si es REENTRENAR (útil en tareas programadas).
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from monitoreo.config import DEFAULT_DB, DEFAULT_PERFIL, FASE4_DIR, REPO_ROOT, Settings
from monitoreo import metricas, politica
from monitoreo.politica import fmt_es
from monitoreo.store import MonitorStore

ICONOS = {"OK": "🟢 OK", "VIGILAR": "🟡 VIGILAR", "REENTRENAR": "🔴 REENTRENAR"}
# Paleta categórica validada (referencia / producción) y tinta de texto
COLOR_REF, COLOR_PROD = "#2a78d6", "#eb6834"
TINTA, TINTA_2, SUPERFICIE, REJILLA = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def _fmt(value, decimales: int = 2, sufijo: str = "") -> str:
    return "—" if value is None else fmt_es(value, decimales) + sufijo


def _pct(value, decimales: int = 1) -> str:
    return "—" if value is None else fmt_es(value * 100, decimales) + " %"


def _tabla(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _etiquetas_bins(cortes: list[float], entero: bool) -> list[str]:
    """Intervalos (a, b]: para años enteros se muestran como a+1–b (o un solo año)."""
    if entero:
        labels = [f"≤{cortes[0]:.0f}"]
        labels += [f"{b:.0f}" if a + 1 == b else f"{a + 1:.0f}–{b:.0f}"
                   for a, b in zip(cortes[:-1], cortes[1:])]
        return labels + [f"≥{cortes[-1] + 1:.0f}"]
    fmt = lambda x: f"{x / 1000:.0f}k"  # noqa: E731
    labels = [f"≤{fmt(cortes[0])}"]
    labels += [f"{fmt(a)}–{fmt(b)}" for a, b in zip(cortes[:-1], cortes[1:])]
    return labels + [f">{fmt(cortes[-1])}"]


# ---------------------------------- figura ----------------------------------
def figura_deriva(deriva: dict, path: Path) -> bool:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variables = {k: v for k, v in deriva["variables"].items() if v.get("psi") is not None}
    if not variables:
        return False
    titulos = {"Year": "Año", "Kilometres": "Kilómetros", "precio_predicho": "Precio (AUD)"}
    fig, axes = plt.subplots(len(variables), 1, figsize=(9, 2.9 * len(variables)), facecolor=SUPERFICIE)
    axes = np.atleast_1d(axes)
    for ax, (name, var) in zip(axes, variables.items()):
        ref = np.array(var["proporciones_referencia"]) * 100
        prod = np.array(var["proporciones_produccion"]) * 100
        x = np.arange(len(ref))
        width = 0.38
        ax.bar(x - width / 2 - 0.01, ref, width, color=COLOR_REF, label="Referencia (entrenamiento)")
        ax.bar(x + width / 2 + 0.01, prod, width, color=COLOR_PROD, label="Producción")
        ax.set_xticks(x, _etiquetas_bins(var["cortes"], entero=name == "Year"), fontsize=8, color=TINTA_2)
        ax.set_ylabel("% de registros", fontsize=9, color=TINTA_2)
        ax.set_title(f"{titulos.get(name, name)} · PSI = {fmt_es(var['psi'], 3)}", loc="left",
                     fontsize=11, color=TINTA)
        ax.set_facecolor(SUPERFICIE)
        ax.grid(axis="y", color=REJILLA, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(REJILLA)
        ax.tick_params(axis="y", labelsize=8, colors=TINTA_2, length=0)
        ax.tick_params(axis="x", length=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, fontsize=9,
               labelcolor=TINTA)
    fig.suptitle("Distribución por deciles de referencia", x=0.01, ha="left", fontsize=12, color=TINTA)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130, facecolor=SUPERFICIE)
    plt.close(fig)
    return True


# ---------------------------------- reporte ----------------------------------
def generar_markdown(r: dict, figura: str | None, enlace_politica: str) -> str:
    pol, op, pr, rend, der = r["politica"], r["operacion"], r["predicciones"], r["rendimiento"], r["deriva"]
    out = [
        "# Reporte de monitoreo del modelo",
        "",
        f"- **Generado (UTC):** {r['generado_utc']}",
        f"- **Ventana:** {r['ventana']['primera_solicitud'] or '—'} → {r['ventana']['ultima_solicitud'] or '—'}",
        f"- **Modelo activo:** `{r['modelo_activo']}`",
        f"- **Decisión de la política:** **{ICONOS[pol['decision']]}**",
        "",
        "## 1. Política de reentrenamiento",
        "",
        _tabla(["Nivel", "Regla", "Motivo"],
               [[ICONOS[m["nivel"]], m["regla"], m["mensaje"]] for m in pol["motivos"]] or [["🟢 OK", "—", "Sin alertas"]]),
        "",
        f"Umbrales y procedimiento: [POLITICA_REENTRENAMIENTO.md]({enlace_politica}).",
        "",
        "## 2. Operación del servicio",
        "",
        _tabla(["Métrica", "Valor"], [
            ["Solicitudes", _fmt(op["total"], 0)],
            ["Por ruta", ", ".join(f"`{k}`: {_fmt(v, 0)}" for k, v in op["por_ruta"].items()) or "—"],
            ["Por código HTTP", ", ".join(f"{k}: {_fmt(v, 0)}" for k, v in op["por_estado"].items()) or "—"],
            ["Errores de cliente (4xx)", _pct(op["tasa_error_cliente"])],
            ["Errores de servidor (5xx)", _pct(op["tasa_error_servidor"])],
            ["Latencia /predict p50 · p95 · máx (ms)",
             " · ".join(_fmt(op["latencia_predict_ms"][k], 1) for k in ("p50", "p95", "max"))],
        ]),
        "",
        "## 3. Predicciones registradas",
        "",
        _tabla(["Métrica", "Valor"], [
            ["Vehículos predichos", _fmt(pr["total"], 0)],
            ["Por versión", ", ".join(f"`{k}`: {_fmt(v, 0)}" for k, v in pr["por_version"].items()) or "—"],
            ["Precio medio (AUD)", _fmt(pr["precio_aud"]["media"])],
            ["P05 · mediana · P95 (AUD)",
             " · ".join(_fmt(pr["precio_aud"][k], 0) for k in ("p05", "mediana", "p95"))],
        ]),
        "",
        "## 4. Rendimiento con precio real",
        "",
        f"Predicciones con precio real recibido por `POST /feedback`: "
        f"**{_fmt(rend['predicciones_con_precio_real'], 0)}** ({_pct(rend['cobertura'], 0)} del total).",
        "",
    ]
    if rend["por_version"]:
        out.append(_tabla(
            ["Versión", "n", "MAE producción (AUD)", "MAE prueba (AUD)", "Razón MAE",
             "MAPE producción", "MAPE prueba", "Razón MAPE"],
            [[f"`{v}`", _fmt(d["n"], 0), _fmt(d["MAE"]), _fmt(d["MAE_prueba"]), _fmt(d["ratio_mae"], 2, "×"),
              _pct(d["MAPE"]), _pct(d["MAPE_prueba"]), _fmt(d["ratio_mape"], 2, "×")]
             for v, d in rend["por_version"].items()],
        ))
        out += ["", "Razón = métrica en producción / métrica en la partición de prueba. "
                "El MAE (AUD) depende del precio de los vehículos; el MAPE es relativo."]
    else:
        out.append("_Aún no hay precios reales._")
    out += ["", "## 5. Deriva de datos", ""]
    if der is None:
        out.append("_No hay perfil de referencia (`python monitor.py perfil`)._")
    else:
        rows = [[name, _fmt(v["n"], 0), _fmt(v.get("psi"), 3),
                 _fmt(v.get("media_referencia"), 1), _fmt(v.get("media_produccion"), 1),
                 _pct(v.get("faltantes"))]
                for name, v in der["variables"].items()]
        out += [
            _tabla(["Variable", "n", "PSI", "Media referencia", "Media producción", "Faltantes"], rows),
            "",
            "PSI < 0,10 estable · 0,10–0,25 cambio moderado · > 0,25 cambio fuerte.",
            f"Marcas no vistas en entrenamiento: {_pct(der['marcas_desconocidas']['proporcion'])}"
            + (f" ({', '.join(der['marcas_desconocidas']['ejemplos'])})" if der["marcas_desconocidas"]["ejemplos"] else ""),
            "",
        ]
        if figura:
            out += [f"![Distribución de referencia frente a producción]({figura})", ""]
    out += [
        "## 6. Versiones del modelo",
        "",
        _tabla(["Versión", "Activa", "Creada (UTC)", "Semilla", "MAE prueba", "R² prueba", "Commit"],
               [[f"`{v['version']}`", "✅" if v["activa"] else "", (v["creado_utc"] or "—")[:19],
                 v["semilla"], _fmt(v["MAE_prueba"]), _fmt(v["R2_prueba"], 3),
                 f"`{v['git_commit'][:7]}`" if v["git_commit"] else "—"] for v in r["versiones"]]),
        "",
    ]
    if r.get("historial"):
        out += ["Historial de activaciones:", "",
                _tabla(["Fecha (UTC)", "Versión", "Anterior"],
                       [[h["ts"][:19], f"`{h['version']}`", f"`{json.loads(h['detalle']).get('anterior')}`"]
                        for h in r["historial"]]), ""]
    return "\n".join(out)


def cmd_reporte(args) -> int:
    settings = Settings.from_env()
    if not Path(args.db).is_file():
        print(f"ERROR: no existe la base de monitoreo {args.db}", file=sys.stderr)
        return 1
    store = MonitorStore(args.db)
    activa = store.active_version() or settings.default_version
    desde = None
    if args.horas:
        desde = (datetime.now(timezone.utc) - timedelta(hours=args.horas)).isoformat(timespec="milliseconds")
    resumen = metricas.calcular_resumen(store, Path(args.models_dir or settings.models_dir), activa,
                                        metricas.cargar_perfil(args.perfil), desde)
    resumen["historial"] = store.events().to_dict(orient="records")

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    figura = "deriva.png" if resumen["deriva"] and figura_deriva(resumen["deriva"], salida / "deriva.png") else None
    enlace = Path(os.path.relpath(FASE4_DIR / "POLITICA_REENTRENAMIENTO.md", salida.resolve())).as_posix()
    (salida / "reporte_monitoreo.md").write_text(generar_markdown(resumen, figura, enlace), encoding="utf-8")
    (salida / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Decisión: {resumen['politica']['decision']}")
    for m in resumen["politica"]["motivos"]:
        print(f"  [{m['nivel']}] {m['regla']}: {m['mensaje']}")
    print(f"Reporte: {(salida / 'reporte_monitoreo.md').resolve()}")
    return 2 if resumen["politica"]["decision"] == "REENTRENAR" else 0


def cmd_perfil(args) -> int:
    perfil = metricas.construir_perfil(Path(args.data))
    Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
    Path(args.salida).write_text(json.dumps(perfil, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Perfil de referencia ({perfil['filas']:,} filas) guardado en {Path(args.salida).resolve()}")
    return 0


def evaluar_versiones(data: pd.DataFrame, versiones: list[str], models_dir: Path) -> pd.DataFrame:
    """MAE/MAPE de cada versión sobre el mismo conjunto etiquetado."""
    from app.registry import ModelLoadError, ModelNotFoundError, ModelRegistry  # fase-3

    registry = ModelRegistry(models_dir)
    y = pd.to_numeric(data["Price"], errors="coerce")
    data, y = data.loc[y.notna() & (y >= 500)], y[y.notna() & (y >= 500)]
    rows = []
    for version in versiones:
        try:
            loaded = registry.get(version)
        except (ModelNotFoundError, ModelLoadError) as exc:
            raise ValueError(str(exc)) from exc
        pred = loaded.model.predict(data.reindex(columns=loaded.metadata["input_columns"]))
        err = np.abs(pred - y.to_numpy())
        rows.append({"version": version, "n": len(y), "MAE": err.mean(),
                     "MAPE": float(np.mean(err / y.to_numpy()))})
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def cmd_comparar(args) -> int:
    settings = Settings.from_env()
    data = pd.read_csv(args.data)
    if "Price" not in data.columns:
        print("ERROR: el CSV de comparación debe incluir la columna Price", file=sys.stderr)
        return 1
    tabla = evaluar_versiones(data, args.versiones, Path(args.models_dir or settings.models_dir))
    print(tabla.to_string(index=False, float_format=lambda x: fmt_es(x, 4)))
    print("AVISO: la comparación solo es válida si ninguna fila del CSV se usó para entrenar "
          "alguna de las versiones comparadas (ver POLITICA_REENTRENAMIENTO.md, sección 5).")
    if args.activa and args.activa in set(tabla["version"]):
        mae_activa = float(tabla.loc[tabla["version"] == args.activa, "MAE"].iloc[0])
        mejor = tabla.iloc[0]
        mejora = 1 - mejor["MAE"] / mae_activa
        if mejor["version"] != args.activa and mejora >= politica.MEJORA_MINIMA_PROMOCION:
            print(f"Recomendación: promover {mejor['version']} (MAE {_pct(mejora)} menor que {args.activa}).")
        else:
            print(f"Recomendación: mantener {args.activa} (ninguna versión mejora el MAE en "
                  f"{_pct(politica.MEJORA_MINIMA_PROMOCION, 0)} o más).")
    return 0


def exportar_etiquetadas(store: MonitorStore, desde: str | None = None) -> pd.DataFrame:
    """Predicciones con precio real, con el esquema del dataset original (columnas + Price)."""
    pred = store.predictions(desde).dropna(subset=["precio_real"])
    if pred.empty:
        return pd.DataFrame()
    filas = pd.DataFrame([json.loads(e) for e in pred["entrada"]])
    return filas.assign(
        Price=pred["precio_real"].to_numpy(),
        prediccion=pred["prediccion"].to_numpy(),
        version=pred["version"].to_numpy(),
        ts=pred["ts"].to_numpy(),
    )


def cmd_exportar(args) -> int:
    if not Path(args.db).is_file():
        print(f"ERROR: no existe la base de monitoreo {args.db}", file=sys.stderr)
        return 1
    desde = None
    if args.horas:
        desde = (datetime.now(timezone.utc) - timedelta(hours=args.horas)).isoformat(timespec="milliseconds")
    data = exportar_etiquetadas(MonitorStore(args.db), desde)
    if data.empty:
        print("No hay predicciones con precio real para exportar.")
        return 0
    Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.salida, index=False)
    print(f"{len(data):,} vehículos con precio real exportados a {Path(args.salida).resolve()}")
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitoreo del modelo de precios de vehículos.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("reporte", help="Genera el reporte de monitoreo")
    p.add_argument("--db", default=str(DEFAULT_DB), help="Base SQLite de la API.")
    p.add_argument("--perfil", default=str(DEFAULT_PERFIL), help="Perfil de referencia (JSON).")
    p.add_argument("--models-dir", default=None, help="Carpeta de versiones (por defecto fase-2/models).")
    p.add_argument("--horas", type=float, default=None, help="Analizar solo las últimas N horas.")
    p.add_argument("--salida", default="reportes", help="Carpeta del reporte.")
    p.set_defaults(func=cmd_reporte)

    p = sub.add_parser("perfil", help="Construye el perfil de referencia desde el dataset")
    p.add_argument("--data", default=str(REPO_ROOT / "data" / "raw" / "australian_vehicle_prices.csv"))
    p.add_argument("--salida", default=str(DEFAULT_PERFIL))
    p.set_defaults(func=cmd_perfil)

    p = sub.add_parser("comparar", help="Compara versiones sobre un CSV etiquetado (con Price)")
    p.add_argument("--data", required=True)
    p.add_argument("--versiones", nargs="+", required=True)
    p.add_argument("--activa", default=None, help="Versión activa, para recomendar o no la promoción.")
    p.add_argument("--models-dir", default=None)
    p.set_defaults(func=cmd_comparar)

    p = sub.add_parser("exportar", help="Exporta las predicciones con precio real a CSV")
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--horas", type=float, default=None, help="Solo las últimas N horas.")
    p.add_argument("--salida", required=True, help="CSV de salida (esquema del dataset + Price).")
    p.set_defaults(func=cmd_exportar)
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    try:
        sys.exit(args.func(args))
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
