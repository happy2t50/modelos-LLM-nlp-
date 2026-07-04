"""
app/config.py — Configuración del microservicio (variables de entorno).
"""

import os
from pathlib import Path

_DIR_BASE = Path(__file__).resolve().parent.parent

# Servidor
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
PREFIJO_API = "/api/v1"

# Base de datos del historial de inferencias (separada del almacén del corpus)
RUTA_DB_INFERENCIAS = _DIR_BASE / "datos" / "inferencias.db"

# Ollama / LLM: generador.py ya lee OLLAMA_URL, QWEN_MODELO, OLLAMA_TIMEOUT.
# Aquí solo derivamos la URL base para el chequeo de readiness.
_URL_GEN = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
URL_OLLAMA_TAGS = _URL_GEN.rsplit("/api/", 1)[0] + "/api/tags"
