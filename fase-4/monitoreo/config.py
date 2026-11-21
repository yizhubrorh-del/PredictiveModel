"""Rutas y configuración de la Fase 4.

No se duplica código: se importa la API de la Fase 3 (que a su vez importa
train.py, src/ y los modelos de la Fase 2). Este módulo agrega fase-3 al sys.path.
"""
import os
import sys
from pathlib import Path

FASE4_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = FASE4_DIR.parent
FASE3_DIR = Path(os.getenv("FASE3_DIR", REPO_ROOT / "fase-3")).resolve()

if str(FASE3_DIR) not in sys.path:
    sys.path.insert(0, str(FASE3_DIR))

from app.config import FASE2_DIR, Settings  # noqa: E402  (fase-3; agrega fase-2 al sys.path)

DEFAULT_DB = Path(os.getenv("MONITOR_DB", FASE4_DIR / "logs" / "monitoreo.db"))
DEFAULT_PERFIL = Path(os.getenv(
    "PERFIL_REFERENCIA", FASE4_DIR / "referencia" / "perfil_referencia.json",
))

__all__ = ["FASE2_DIR", "FASE3_DIR", "FASE4_DIR", "REPO_ROOT", "Settings",
           "DEFAULT_DB", "DEFAULT_PERFIL"]
