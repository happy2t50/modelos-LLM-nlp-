"""
app/main.py — Microservicio de diagnóstico agrícola (modo online).

Expone el pipeline RAG (fusión + recuperación + LLM) como API REST para la app
móvil AgroGraph-MAS. La CNN corre en el dispositivo; el servicio recibe su
resultado y genera el diagnóstico.

Ejecutar:
    pip install fastapi "uvicorn[standard]"
    set PYTHONUTF8=1
    uvicorn app.main:app --host 0.0.0.0 --port 8000
    # Documentación Swagger en http://localhost:8000/docs
"""

import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import base64

import requests
from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.concurrency import run_in_threadpool

from app import config
from app.schemas import (
    ConsultaRequest, LlmResponse, HistorialResponse, InferenciaResumen,
    CatalogResponse, DocumentDownloadResponse, MapaCampaniasResponse, AlertaResponse,
)


def _rol_desde_jwt(authorization: str | None) -> str | None:
    """
    Lee el claim 'rol' de un JWT 'Bearer <token>' SIN verificar la firma.
    Stopgap hasta que exista auth real: el rol solo cambia el TONO de la
    respuesta (no da acceso), por lo que leerlo sin verificar es de bajo riesgo.
    Devuelve None si no hay token o no trae 'rol'.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    try:
        token = authorization.split(" ", 1)[1]
        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)  # padding base64url
        datos = json.loads(base64.urlsafe_b64decode(payload_b64))
        rol = datos.get("rol") or datos.get("role")
        # normaliza 'aprendiz_agricola' → 'aprendiz'
        if rol and "aprendiz" in str(rol).lower():
            return "aprendiz"
        if rol and "agricultor" in str(rol).lower():
            return "agricultor"
        return rol
    except Exception:
        return None
from app import servicio, db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warmup: precarga BERT y calienta Ollama (evita 503 por arranque en frío)."""
    try:
        from modulos.busqueda_semantica import _obtener_modelo
        _obtener_modelo()
        print("[api] Modelo BERT precargado.")
    except Exception as e:
        print(f"[api] Aviso: no se pudo precargar BERT: {e}")

    # Calentar Ollama: cargar el modelo en memoria con un prompt trivial.
    try:
        import os
        url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
        modelo = os.environ.get("QWEN_MODELO", "qwen3.5:0.8b")
        requests.post(url, json={"model": modelo, "prompt": "ok",
                                 "stream": False, "think": False},
                      timeout=90)
        print("[api] Ollama precalentado.")
    except Exception as e:
        print(f"[api] Aviso: no se pudo precalentar Ollama: {e}")
    yield


app = FastAPI(
    title="Microservicio de diagnóstico agrícola (RAG + LLM)",
    description="Recibe el resultado de la CNN + texto y genera el diagnóstico "
                "con documentos (RAG) y Qwen. Contrato compatible con la app móvil.",
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────
# Inferencia
# ─────────────────────────────────────────────

@app.post(config.PREFIJO_API + "/consultar", response_model=LlmResponse,
          tags=["inferencia"], summary="Diagnóstico (CNN result + texto → RAG + LLM)")
async def consultar_endpoint(req: ConsultaRequest,
                             authorization: str | None = Header(None)):
    # El rol viene del JWT si está presente (como espera la app); si no, del body.
    rol_jwt = _rol_desde_jwt(authorization)
    if rol_jwt:
        req.rol = rol_jwt
    t0 = time.time()
    try:
        resultado = await run_in_threadpool(servicio.ejecutar_consulta, req)
    except RuntimeError as e:
        # generador._llamar_ollama lanza RuntimeError si Ollama no responde/timeout
        raise HTTPException(status_code=503, detail=f"LLM no disponible: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")

    latency_ms = int((time.time() - t0) * 1000)
    inf_id, creado = db.guardar_inferencia(
        req.model_dump(), resultado, resultado["modo"], latency_ms)

    resultado["inference_id"] = inf_id
    return resultado


# ─────────────────────────────────────────────
# Historial
# ─────────────────────────────────────────────

@app.get(config.PREFIJO_API + "/inferences", response_model=HistorialResponse,
         tags=["historial"], summary="Listar inferencias (paginado)")
async def historial_endpoint(limit: int = Query(20, ge=1, le=100),
                             offset: int = Query(0, ge=0)):
    return db.listar_inferencias(limit=limit, offset=offset)


@app.get(config.PREFIJO_API + "/inferences/{inference_id}",
         tags=["historial"], summary="Detalle de una inferencia")
async def detalle_endpoint(inference_id: str):
    detalle = db.obtener_inferencia(inference_id)
    if detalle is None:
        raise HTTPException(status_code=404, detail="Inferencia no encontrada")
    return detalle


# ─────────────────────────────────────────────
# Mapa epidemiológico REAL (campañas fitosanitarias SENASICA)
# ─────────────────────────────────────────────

@app.get(config.PREFIJO_API + "/clustering/mapa-campanias",
         response_model=MapaCampaniasResponse, tags=["clustering"],
         summary="Mapa epidemiológico REAL (campañas SENASICA por estado)")
async def clustering_mapa_campanias():
    from app import campanias
    return await run_in_threadpool(campanias.mapa)


@app.get(config.PREFIJO_API + "/alertas", response_model=AlertaResponse,
         tags=["clustering"],
         summary="Alerta epidemiológica real (campaña dominante por estado)")
async def alerta_epidemiologica(estado: str | None = Query(
        None, description="Entidad federativa; si se omite, alerta nacional")):
    from app import campanias
    return await run_in_threadpool(campanias.alerta, estado)


# ─────────────────────────────────────────────
# Offline: catálogo y descarga de documentos (para RAG on-device)
# ─────────────────────────────────────────────

@app.get(config.PREFIJO_API + "/offline/catalog", response_model=CatalogResponse,
         tags=["offline"], summary="Catálogo de documentos descargables")
async def offline_catalog():
    from app import offline
    return offline.catalogo()


@app.get(config.PREFIJO_API + "/offline/documents/{doc_id}",
         response_model=DocumentDownloadResponse, tags=["offline"],
         summary="Descargar un documento con sus chunks y embeddings (384-d)")
async def offline_document(doc_id: str):
    from app import offline
    doc = await run_in_threadpool(offline.documento, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Documento no encontrado")
    return doc


# ─────────────────────────────────────────────
# Salud
# ─────────────────────────────────────────────

@app.get("/health", tags=["salud"], summary="Liveness")
async def health():
    return {"status": "ok"}


@app.get("/ready", tags=["salud"], summary="Readiness (comprueba Ollama)")
async def ready():
    try:
        r = requests.get(config.URL_OLLAMA_TAGS, timeout=3)
        ollama_ok = r.status_code == 200
    except Exception:
        ollama_ok = False
    if not ollama_ok:
        raise HTTPException(status_code=503, detail="Ollama no disponible")
    return {"status": "ready", "ollama": True}
