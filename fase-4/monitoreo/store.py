"""Registro persistente (SQLite) de solicitudes, predicciones y eventos del modelo.

Se abre una conexión por operación: es seguro con los hilos de FastAPI y con
scripts que leen la misma base mientras la API escribe (modo WAL).
"""
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS solicitudes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    metodo TEXT NOT NULL,
    ruta TEXT NOT NULL,
    estado INTEGER NOT NULL,
    latencia_ms REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS predicciones (
    id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    lote TEXT NOT NULL,
    version TEXT NOT NULL,
    prediccion REAL NOT NULL,
    brand TEXT,
    year INTEGER,
    kilometres REAL,
    entrada TEXT NOT NULL,
    precio_real REAL,
    ts_precio_real TEXT
);
CREATE TABLE IF NOT EXISTS eventos_modelo (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    evento TEXT NOT NULL,
    version TEXT NOT NULL,
    detalle TEXT
);
CREATE INDEX IF NOT EXISTS idx_solicitudes_ts ON solicitudes(ts);
CREATE INDEX IF NOT EXISTS idx_predicciones_ts ON predicciones(ts);
"""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class FeedbackError(Exception):
    """Precio real para una predicción inexistente o ya etiquetada."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


class MonitorStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=10)
        try:
            with conn:  # commit o rollback automático
                yield conn
        finally:
            conn.close()

    # ------------------------------- escritura -------------------------------
    def log_request(self, metodo: str, ruta: str, estado: int, latencia_ms: float) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO solicitudes (ts, metodo, ruta, estado, latencia_ms) VALUES (?, ?, ?, ?, ?)",
                (now_utc(), metodo, ruta, estado, latencia_ms),
            )

    def log_predictions(self, version: str, records: list[dict], predictions: list[float]) -> list[str]:
        ts, lote = now_utc(), uuid.uuid4().hex
        ids = [uuid.uuid4().hex for _ in predictions]
        rows = [
            (pid, ts, lote, version, float(pred), rec.get("Brand"), rec.get("Year"),
             rec.get("Kilometres"), json.dumps(rec, ensure_ascii=False))
            for pid, rec, pred in zip(ids, records, predictions)
        ]
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO predicciones (id, ts, lote, version, prediccion, brand, year, "
                "kilometres, entrada) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return ids

    def add_feedback(self, prediction_id: str, precio_real: float) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT version, prediccion, precio_real FROM predicciones WHERE id = ?",
                (prediction_id,),
            ).fetchone()
            if row is None:
                raise FeedbackError(404, f"No existe la predicción '{prediction_id}'")
            if row[2] is not None:
                raise FeedbackError(409, f"La predicción '{prediction_id}' ya tiene precio real")
            conn.execute(
                "UPDATE predicciones SET precio_real = ?, ts_precio_real = ? WHERE id = ?",
                (precio_real, now_utc(), prediction_id),
            )
        return {"id_prediccion": prediction_id, "version": row[0],
                "prediccion": row[1], "precio_real": precio_real,
                "error_absoluto": round(abs(row[1] - precio_real), 2)}

    def log_event(self, evento: str, version: str, detalle: dict | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO eventos_modelo (ts, evento, version, detalle) VALUES (?, ?, ?, ?)",
                (now_utc(), evento, version, json.dumps(detalle or {}, ensure_ascii=False)),
            )

    # -------------------------------- lectura --------------------------------
    def active_version(self) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT version FROM eventos_modelo WHERE evento = 'activacion' "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else None

    def _frame(self, table: str, since: str | None) -> pd.DataFrame:
        query = f"SELECT * FROM {table}"  # table proviene de constantes internas
        params: tuple = ()
        if since:
            query += " WHERE ts >= ?"
            params = (since,)
        with self._connect() as conn:
            return pd.read_sql_query(query + " ORDER BY ts", conn, params=params)

    def requests(self, since: str | None = None) -> pd.DataFrame:
        return self._frame("solicitudes", since)

    def predictions(self, since: str | None = None) -> pd.DataFrame:
        return self._frame("predicciones", since)

    def events(self) -> pd.DataFrame:
        return self._frame("eventos_modelo", None)
