# Microservicio online (API REST)

Expone el pipeline RAG (fusión + recuperación + LLM) como API para la app móvil.
La **CNN corre en el dispositivo** (TFLite); el servicio recibe su resultado + el
texto y genera el diagnóstico con Qwen (Ollama).

## Requisitos previos

```powershell
pip install -r requirements.txt          # incluye fastapi, uvicorn
python scripts/construir_corpus.py       # construye el corpus/índices (una vez)
ollama pull qwen3.5:0.8b                  # o qwen3.5:4b en la nube
```

## Ejecutar

```powershell
set PYTHONUTF8=1
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- **Swagger / OpenAPI:** http://localhost:8000/docs
- **OpenAPI JSON:** http://localhost:8000/openapi.json

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| POST | `/api/v1/consultar` | Resultado CNN + texto → diagnóstico (RAG + LLM) |
| GET | `/api/v1/inferences` | Historial paginado (`limit`, `offset`) |
| GET | `/api/v1/inferences/{id}` | Detalle de una inferencia |
| GET | `/api/v1/clustering/mapa-campanias` | **Mapa epidemiológico REAL** (campañas SENASICA por estado) |
| GET | `/api/v1/alertas?estado=` | **Alerta epidemiológica real** (campaña/plaga dominante por estado) |
| GET | `/api/v1/offline/catalog` | Catálogo de documentos descargables (para RAG on-device) |
| GET | `/api/v1/offline/documents/{id}` | Documento con chunks + embeddings (384-d) |
| GET | `/health` | Liveness |
| GET | `/ready` | Readiness (comprueba Ollama) |

> El mapa epidemiológico usa **datos reales** de campañas fitosanitarias (SENASICA)
> en `datos/campanias/*.csv`. No hay datos sintéticos en el servicio.

## Contrato de `/api/v1/consultar`

**Request** (JSON):
```json
{
  "cropName": "Calabaza", "diseaseName": "oídio", "confidence": 0.47,
  "texto": "polvo blanco como ceniza", "rol": "agricultor",
  "cultivos": ["calabaza", "maiz"]
}
```

**Response** (compatible con `LlmResponseEntity` de la app Flutter):
```json
{
  "diagnostico": "...", "tratamiento": "...", "prevencion": "...",
  "fuentes": ["..."], "confianzaAjustada": 0.62, "estado": "reforzado",
  "explicacion": "...", "sintomas": ["oidio"], "avisos": [],
  "sinDocumentos": false, "inference_id": "inf_...", "modo": "online"
}
```

Cada inferencia se persiste en `datos/inferencias.db` (SQLite) y se consulta por
el endpoint de historial.

## Configuración (variables de entorno)

`OLLAMA_URL`, `QWEN_MODELO`, `OLLAMA_TIMEOUT` (las lee `modulos/generador.py`);
`HOST`, `PORT` para el servidor. En producción, `OLLAMA_URL` apunta al contenedor
de Ollama en la red privada (p. ej. `http://ollama:11434/api/generate`).
