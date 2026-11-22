"""API de la Fase 3 ampliada con monitoreo y gestión de versiones (Fase 4).

Ejecutar desde fase-4:  uvicorn api_monitoreo:create_app --factory --port 8000

Mantiene /health, /predict y /train de la Fase 3 (importados, no copiados) y agrega:
  - registro de cada solicitud y de cada predicción (SQLite),
  - POST /feedback        precio real de una predicción ya hecha,
  - GET  /metrics         métricas de operación, predicciones, rendimiento y deriva,
  - GET  /models          versiones con sus métricas e historial de activaciones,
  - PUT  /models/active   cambia la versión que usa /predict (promoción o reversión).
"""
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from monitoreo.config import DEFAULT_DB, DEFAULT_PERFIL, Settings  # agrega fase-3 al sys.path
from app.main import APIError, create_app as create_base_app  # fase-3
from app.registry import ModelLoadError, ModelNotFoundError  # fase-3
from app.schemas import VERSION_PATTERN  # fase-3
from monitoreo import metricas
from monitoreo.store import FeedbackError, MonitorStore

logger = logging.getLogger("api.monitoreo")
NO_REGISTRAR = ("/docs", "/redoc", "/openapi.json", "/favicon.ico")


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id_prediccion: str = Field(pattern=r"^[0-9a-f]{32}$")
    precio_real: float = Field(gt=0, le=10_000_000, description="Precio real de venta (AUD)")


class ActiveVersionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str = Field(pattern=VERSION_PATTERN)


def create_app(settings: Settings | None = None, db_path: Path | None = None,
               perfil_path: Path | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = MonitorStore(db_path or DEFAULT_DB)
    perfil = metricas.cargar_perfil(perfil_path or DEFAULT_PERFIL)

    def active_version() -> str:
        return store.active_version() or settings.default_version

    def on_prediction(version, records, predictions, _latency_ms):
        try:
            return store.log_predictions(version, records, predictions)
        except Exception:  # el monitoreo nunca debe impedir responder la predicción
            logger.exception("No se pudo registrar la predicción")
            return None

    app = create_base_app(settings, active_version=active_version, on_prediction=on_prediction)
    app.title = "API de precios de vehículos con monitoreo"
    app.version = "4.0.0"
    app.description = "Fase 4: API de la Fase 3 con registro de predicciones, métricas y versiones."
    app.state.store = store
    app.state.perfil = perfil
    registry = app.state.registry

    @app.middleware("http")
    async def registrar_solicitud(request: Request, call_next):
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            path = request.url.path
            if not path.startswith(NO_REGISTRAR):
                latency = (time.perf_counter() - start) * 1000
                try:
                    await run_in_threadpool(store.log_request, request.method, path, status, latency)
                except Exception:
                    logger.exception("No se pudo registrar la solicitud")

    @app.post("/feedback", tags=["monitoreo"], status_code=201,
              responses={404: {"description": "Predicción inexistente"},
                         409: {"description": "La predicción ya tiene precio real"}})
    def feedback(request: FeedbackRequest):
        """Registra el precio real de una predicción (permite medir el error en producción)."""
        try:
            return store.add_feedback(request.id_prediccion, request.precio_real)
        except FeedbackError as exc:
            raise APIError(exc.status_code, str(exc)) from exc

    @app.get("/metrics", tags=["monitoreo"])
    def metrics(horas: float | None = Query(default=None, gt=0, le=24 * 365,
                                            description="Ventana: últimas N horas (por defecto todo).")):
        """Métricas de operación, predicciones, rendimiento real, deriva y decisión de la política."""
        desde = None
        if horas is not None:
            desde = (datetime.now(timezone.utc) - timedelta(hours=horas)).isoformat(timespec="milliseconds")
        return metricas.calcular_resumen(store, settings.models_dir, active_version(), perfil, desde)

    @app.get("/models", tags=["versiones"])
    def models():
        """Versiones disponibles, versión activa e historial de activaciones."""
        events = store.events()
        return {
            "activa": active_version(),
            "versiones": metricas.info_versiones(settings.models_dir, active_version()),
            "historial": events.drop(columns=["id"]).to_dict(orient="records"),
        }

    @app.put("/models/active", tags=["versiones"],
             responses={404: {"description": "Versión inexistente"}})
    def set_active(request: ActiveVersionRequest):
        """Activa una versión existente (promoción de un modelo nuevo o reversión)."""
        anterior = active_version()
        try:
            registry.get(request.version)  # comprueba que existe y carga antes de activarla
        except ModelNotFoundError as exc:
            raise APIError(404, f"{exc}. Disponibles: {registry.available_versions()}") from exc
        except ModelLoadError as exc:
            raise APIError(500, str(exc)) from exc
        store.log_event("activacion", request.version, {"anterior": anterior})
        return {"activa": request.version, "anterior": anterior}

    return app

