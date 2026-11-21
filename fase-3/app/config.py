"""Configuración de la API, leída de variables de entorno.

La API no duplica código: reutiliza `train.py`, `src/` y los modelos versionados
de la Fase 2. Este módulo agrega fase-2 al `sys.path` para poder importarlos
(los .joblib también referencian `src.features.clean_raw`).
"""
import os
import sys
from dataclasses import dataclass
from pathlib import Path

FASE3_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = FASE3_DIR.parent
FASE2_DIR = Path(os.getenv("FASE2_DIR", REPO_ROOT / "fase-2")).resolve()

if str(FASE2_DIR) not in sys.path:
    sys.path.insert(0, str(FASE2_DIR))


@dataclass(frozen=True)
class Settings:
    models_dir: Path
    data_path: Path
    default_version: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            models_dir=Path(os.getenv("MODELS_DIR", FASE2_DIR / "models")),
            data_path=Path(os.getenv(
                "DATA_PATH", REPO_ROOT / "data" / "raw" / "australian_vehicle_prices.csv",
            )),
            default_version=os.getenv("DEFAULT_MODEL_VERSION", "v1"),
        )
