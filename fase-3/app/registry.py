"""Carga y caché de las versiones del modelo guardadas por train.py."""
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import joblib

from app.schemas import VERSION_PATTERN

MODEL_FILE = "modelo.joblib"
METADATA_FILE = "metadata.json"


class ModelNotFoundError(Exception):
    """La versión solicitada no existe en el directorio de modelos."""


class ModelLoadError(Exception):
    """La versión existe pero sus artefactos no se pueden cargar."""


@dataclass(frozen=True)
class LoadedModel:
    version: str
    model: object
    metadata: dict


class ModelRegistry:
    """Las versiones no se sobrescriben (train.py lo impide), así que la caché es segura."""

    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self._cache: dict[str, LoadedModel] = {}
        self._lock = threading.Lock()

    def available_versions(self) -> list[str]:
        if not self.models_dir.is_dir():
            return []
        return sorted(
            p.name for p in self.models_dir.iterdir()
            if (p / MODEL_FILE).is_file() and (p / METADATA_FILE).is_file()
        )

    def is_loaded(self, version: str) -> bool:
        return version in self._cache

    def get(self, version: str) -> LoadedModel:
        if not re.fullmatch(VERSION_PATTERN, version):
            raise ModelNotFoundError(f"Identificador de versión inválido: '{version}'")
        with self._lock:
            if version not in self._cache:
                self._cache[version] = self._load(version)
            return self._cache[version]

    def _load(self, version: str) -> LoadedModel:
        version_dir = self.models_dir / version
        model_path = version_dir / MODEL_FILE
        metadata_path = version_dir / METADATA_FILE
        if not model_path.is_file() or not metadata_path.is_file():
            raise ModelNotFoundError(f"No existe la versión del modelo '{version}'")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelLoadError(f"metadata.json de '{version}' no es válido: {exc}") from exc
        if not isinstance(metadata.get("input_columns"), list):
            raise ModelLoadError(f"metadata.json de '{version}' no define 'input_columns'")
        try:
            # Solo se cargan artefactos propios: joblib no es seguro con archivos de terceros.
            model = joblib.load(model_path)
        except Exception as exc:
            raise ModelLoadError(f"No se pudo cargar el modelo '{version}': {exc}") from exc
        return LoadedModel(version=version, model=model, metadata=metadata)
