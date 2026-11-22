"""Política de reentrenamiento (ver POLITICA_REENTRENAMIENTO.md).

Cada regla produce un nivel: OK < VIGILAR < REENTRENAR. La decisión global es el
nivel más alto. Los umbrales están en un solo lugar para que el documento y el
código no diverjan.
"""
from datetime import datetime, timezone

# Mínimos y razones calibrados con submuestras de la partición de prueba de v1
# (sin deriva real): con n = 400, el PSI por azar llega a 0,053 y la razón de
# MAE a 1,257 en el percentil 99. Ver la sección 4 de POLITICA_REENTRENAMIENTO.md.
UMBRALES = {
    # R1. Deriva de datos (PSI)
    "min_predicciones_deriva": 400,
    "psi_vigilar": 0.10,
    "psi_reentrenar": 0.25,
    "marcas_desconocidas_vigilar": 0.05,
    # R2. Rendimiento con precio real: métrica en producción / métrica de prueba
    "min_precios_reales": 400,
    "ratio_vigilar": 1.20,
    "ratio_reentrenar": 1.30,
    # R3. Antigüedad del modelo activo
    "dias_max_modelo": 180,
    # R4. Operación (no se corrige reentrenando)
    "tasa_error_servidor_max": 0.01,
    "latencia_p95_max_ms": 1000,
}

NIVELES = ["OK", "VIGILAR", "REENTRENAR"]

# Promoción: una versión candidata reemplaza a la activa solo si, evaluadas sobre
# los mismos datos etiquetados que ninguna vio al entrenar, reduce el MAE al menos
# en este porcentaje.
MEJORA_MINIMA_PROMOCION = 0.02


def fmt_es(value: float, decimales: int = 2) -> str:
    """Formato numérico del proyecto: 4.929,86."""
    texto = f"{value:,.{decimales}f}"
    return texto.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def _dias_desde(iso: str | None) -> float | None:
    if not iso:
        return None
    created = datetime.fromisoformat(iso)
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - created).total_seconds() / 86400


def _nivel_ratio(ratio: float, u: dict) -> str:
    if ratio > u["ratio_reentrenar"]:
        return "REENTRENAR"
    return "VIGILAR" if ratio > u["ratio_vigilar"] else "OK"


def evaluar(resumen: dict, umbrales: dict = UMBRALES) -> dict:
    u = umbrales
    motivos: list[dict] = []

    def add(nivel: str, regla: str, mensaje: str):
        motivos.append({"nivel": nivel, "regla": regla, "mensaje": mensaje})

    # R1. Deriva de los datos de entrada y de la salida del modelo
    deriva = resumen.get("deriva")
    n_pred = resumen["predicciones"]["total"]
    if deriva is None:
        add("VIGILAR", "R1-deriva", "No hay perfil de referencia: no se puede medir la deriva.")
    elif n_pred < u["min_predicciones_deriva"]:
        add("OK", "R1-deriva", f"Solo {n_pred} predicciones; se requieren "
            f"{u['min_predicciones_deriva']} para evaluar la deriva.")
    else:
        valores = {k: v["psi"] for k, v in deriva["variables"].items() if v.get("psi") is not None}
        for name, value in valores.items():
            if value > u["psi_reentrenar"]:
                add("REENTRENAR", "R1-deriva",
                    f"PSI de {name} = {fmt_es(value, 3)} > {fmt_es(u['psi_reentrenar'])}.")
            elif value > u["psi_vigilar"]:
                add("VIGILAR", "R1-deriva",
                    f"PSI de {name} = {fmt_es(value, 3)} > {fmt_es(u['psi_vigilar'])}.")
        if valores and max(valores.values()) <= u["psi_vigilar"]:
            peor = max(valores, key=valores.get)
            add("OK", "R1-deriva", f"Distribuciones estables: PSI máximo {fmt_es(valores[peor], 3)} "
                f"({peor}) ≤ {fmt_es(u['psi_vigilar'])}.")
        share = deriva["marcas_desconocidas"]["proporcion"]
        if share > u["marcas_desconocidas_vigilar"]:
            add("VIGILAR", "R1-deriva",
                f"{fmt_es(share * 100, 1)} % de los vehículos son de marcas no vistas en entrenamiento.")

    # R2. Rendimiento real del modelo activo (MAE en AUD y MAPE relativo)
    activa = resumen["modelo_activo"]
    perf = resumen["rendimiento"]["por_version"].get(activa)
    if perf is None or perf["n"] < u["min_precios_reales"]:
        n = perf["n"] if perf else 0
        add("OK", "R2-rendimiento", f"{n} precios reales para {activa}; se requieren "
            f"{u['min_precios_reales']} para evaluar el error.")
    else:
        for metrica, unidad in (("MAE", " AUD"), ("MAPE", "")):
            ratio = perf.get(f"ratio_{metrica.lower()}")
            if ratio is None:
                continue
            prod, test = perf[metrica], perf[f"{metrica}_prueba"]
            if metrica == "MAPE":
                prod_txt, test_txt = f"{fmt_es(prod * 100, 1)} %", f"{fmt_es(test * 100, 1)} %"
            else:
                prod_txt, test_txt = fmt_es(prod, 0) + unidad, fmt_es(test, 0) + unidad
            add(_nivel_ratio(ratio, u), "R2-rendimiento",
                f"{metrica} de {activa} en producción = {prod_txt}, {fmt_es(ratio)}× el de prueba "
                f"({test_txt}).")

    # R3. Antigüedad del modelo activo (reentrenamiento periódico)
    info = next((v for v in resumen["versiones"] if v["version"] == activa), None)
    dias = _dias_desde(info["creado_utc"]) if info else None
    if dias is not None:
        nivel = "REENTRENAR" if dias > u["dias_max_modelo"] else "OK"
        add(nivel, "R3-antiguedad", f"El modelo {activa} tiene {dias:.0f} días "
            f"(máximo {u['dias_max_modelo']}).")

    # R4. Salud operativa: se vigila, pero reentrenar no la corrige
    op = resumen["operacion"]
    if op["tasa_error_servidor"] > u["tasa_error_servidor_max"]:
        add("VIGILAR", "R4-operacion",
            f"Errores 5xx = {fmt_es(op['tasa_error_servidor'] * 100, 1)} %: "
            "revisar el servicio (no se corrige reentrenando).")
    p95 = op["latencia_predict_ms"]["p95"]
    if p95 is not None and p95 > u["latencia_p95_max_ms"]:
        add("VIGILAR", "R4-operacion", f"Latencia p95 de /predict = {fmt_es(p95, 0)} ms "
            f"> {u['latencia_p95_max_ms']} ms.")

    decision = max((m["nivel"] for m in motivos), key=NIVELES.index, default="OK")
    motivos.sort(key=lambda m: -NIVELES.index(m["nivel"]))
    return {"decision": decision, "motivos": motivos, "umbrales": umbrales}
