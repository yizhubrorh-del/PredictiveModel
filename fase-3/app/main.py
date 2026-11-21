"""API REST del modelo de precios de vehículos (Fase 3).

Ejecutar desde fase-3:  uvicorn app.main:app --port 8000
Documentación interactiva: http://localhost:8000/docs
"""
import json
import logging
import threading
import time
from argparse import Namespace

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import Settings  # agrega fase-2 al sys.path: debe ir antes de `train`
import train as training  # fase-2/train.py
from app.registry import ModelLoadError, ModelNotFoundError, ModelRegistry
from app.schemas import (
    VERSION_PATTERN,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TrainResponse,
)

logger = logging.getLogger("api")

ERROR_RESPONSES = {
    404: {"description": "Versión del modelo inexistente"},
    422: {"description": "Entrada inválida"},
    500: {"description": "Error interno"},
}


class APIError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message


def _error(status_code: int, message: str, details=None) -> JSONResponse:
    body = {"error": message}
    if details:
        body["detalles"] = details
    return JSONResponse(status_code=status_code, content=body)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    registry = ModelRegistry(settings.models_dir)
    train_lock = threading.Lock()

    app = FastAPI(
        title="API de precios de vehículos (Australia)",
        version="3.0.0",
        description="Fase 3: expone el modelo de las fases 1-2 mediante /health, /predict y /train.",
    )
    app.state.settings = settings
    app.state.registry = registry
    app.state.train_lock = train_lock

    # ---------- Manejo de errores: siempre {"error": ..., "detalles": [...]} ----------
    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError):
        details = [
            {
                "campo": ".".join(str(p) for p in err["loc"] if p != "body"),
                "mensaje": err["msg"],
            }
            for err in exc.errors()
        ]
        return _error(422, "Entrada inválida", details)

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(_: Request, exc: StarletteHTTPException):
        message = {404: "Recurso no encontrado", 405: "Método no permitido"}.get(
            exc.status_code, str(exc.detail),
        )
        return _error(exc.status_code, message)

    @app.exception_handler(APIError)
    async def api_error_handler(_: Request, exc: APIError):
        return _error(exc.status_code, exc.message)

    @app.exception_handler(Exception)
    async def unexpected_handler(_: Request, exc: Exception):
        logger.exception("Error no controlado", exc_info=exc)
        return _error(500, "Error interno del servidor")

    def load(version: str):
        try:
            return registry.get(version)
        except ModelNotFoundError as exc:
            raise APIError(404, f"{exc}. Disponibles: {registry.available_versions()}") from exc
        except ModelLoadError as exc:
            logger.exception("Fallo al cargar el modelo")
            raise APIError(500, str(exc)) from exc

    # ---------------------------------- Endpoints ----------------------------------
    @app.get("/health", response_model=HealthResponse, tags=["estado"])
    def health():
        """Estado del servicio. Responde 503 si el modelo por defecto no se puede cargar."""
        detail = None
        try:
            registry.get(settings.default_version)
        except (ModelNotFoundError, ModelLoadError) as exc:
            detail = str(exc)
        body = HealthResponse(
            status="ok" if detail is None else "degradado",
            modelo_por_defecto=settings.default_version,
            modelo_cargado=registry.is_loaded(settings.default_version),
            versiones_disponibles=registry.available_versions(),
            dataset_disponible=settings.data_path.is_file(),
            detalle=detail,
        )
        return JSONResponse(status_code=200 if detail is None else 503, content=body.model_dump())

    @app.post("/predict", response_model=PredictResponse, responses=ERROR_RESPONSES, tags=["modelo"])
    def predict(
        request: PredictRequest,
        version: str | None = Query(
            default=None, pattern=VERSION_PATTERN,
            description="Versión del modelo; por defecto la configurada (v1).",
        ),
    ):
        """Predice el precio (AUD) de uno o varios vehículos."""
        loaded = load(version or settings.default_version)
        # Mismo orden de columnas que en el entrenamiento (como fase-2/predict.py);
        # los null se tratan como faltantes por clean_raw y los imputadores del pipeline.
        X = pd.DataFrame([v.to_record() for v in request.vehicles]).reindex(
            columns=loaded.metadata["input_columns"],
        )
        predictions = np.asarray(loaded.model.predict(X), dtype=float)
        if not np.isfinite(predictions).all():
            raise APIError(500, "El modelo produjo predicciones no válidas")
        return PredictResponse(
            version=loaded.version,
            n=len(predictions),
            predicciones=[round(float(p), 2) for p in predictions],
        )

    @app.post(
        "/train", response_model=TrainResponse, status_code=201, tags=["modelo"],
        responses={**ERROR_RESPONSES, 409: {"description": "Versión duplicada o entrenamiento en curso"},
                   503: {"description": "Dataset no disponible"}},
    )
    def train_model(request: TrainRequest):
        """Entrena y guarda una versión nueva con el dataset del servidor (síncrono)."""
        if not settings.data_path.is_file():
            raise APIError(503, "El dataset de entrenamiento no está disponible en el servidor")
        if (settings.models_dir / request.version).exists():
            raise APIError(409, f"La versión '{request.version}' ya existe. Usa otro identificador.")
        if not train_lock.acquire(blocking=False):
            raise APIError(409, "Ya hay un entrenamiento en curso. Inténtalo más tarde.")
        try:
            start = time.perf_counter()
            args = Namespace(
                data=str(settings.data_path), target="Price",
                output_dir=str(settings.models_dir), version=request.version,
                seed=request.seed, test_size=request.test_size,
                source_commit=request.source_commit,
            )
            try:
                version_dir = training.train(args)
            except FileExistsError as exc:
                raise APIError(409, str(exc)) from exc
            except (ValueError, pd.errors.ParserError) as exc:
                raise APIError(400, f"No fue posible entrenar: {exc}") from exc
            duration = time.perf_counter() - start
        finally:
            train_lock.release()

        metrics = json.loads((version_dir / "metrics.json").read_text(encoding="utf-8"))
        metadata = json.loads((version_dir / "metadata.json").read_text(encoding="utf-8"))
        return TrainResponse(
            version=request.version,
            metricas={"modelo": metrics["model"], "linea_base": metrics["baseline"]},
            filas_entrenamiento=metadata["data"]["rows_train"],
            filas_prueba=metadata["data"]["rows_test"],
            git_commit=metadata.get("git_commit"),
            duracion_s=round(duration, 2),
        )

    return app


app = create_app()
