# Plan del Microservicio — Diagnóstico Agrícola (API en la nube)

> Documento de arquitectura y plan de implementación para exponer el sistema
> (CNN + NLP + RAG + Qwen) como un microservicio HTTP.
> Versión 1.0 · Alineado con la decisión de despliegue **nube + dispositivo**.

---

## 1. Objetivo y alcance

Convertir el orquestador actual (`modulos/asistente.py`) en un **microservicio HTTP**
que la app móvil consume **cuando hay cobertura** (modo online). El servidor hace el
trabajo pesado (búsqueda híbrida + LLM grande); el móvil solo envía la foto + texto y
muestra la respuesta.

**Dentro del alcance:**
- API REST que recibe imagen + texto + rol + cultivos y devuelve el diagnóstico.
- Carga de modelos (CNN EfficientNet-B4 y BERT) una sola vez al arrancar.
- Generación con Qwen vía Ollama (en la nube puede ser un modelo grande, p. ej. `qwen3.5:4b`).
- Contenerización con Docker y orquestación con docker-compose.

**Fuera del alcance (lo aclara este plan, pero no lo implementa el microservicio):**
- El **modo offline** del móvil (corpus + modelos pequeños a bordo). Eso vive en el
  dispositivo, NO en este servicio. El servicio es siempre "online".
- El login / gestión de usuarios (lo maneja el backend de la app, fuera del RAG).

---

## 2. Dónde encaja (arquitectura general)

```
┌──────────────────────────────────────────────────────────────────┐
│                         APP MÓVIL (cliente)                        │
│                                                                    │
│   ¿hay cobertura?                                                   │
│      │                                                              │
│      ├── SÍ (online) ───────────►  llama a este MICROSERVICIO      │
│      │                              (la nube hace todo)            │
│      │                                                              │
│      └── NO (offline) ──────────►  responde EN EL DISPOSITIVO      │
│                                     (SQLite Top-K + TF-IDF +       │
│                                      BERT pequeño + Qwen 0.8b)     │
└──────────────────────────────────────────────────────────────────┘
                                   │  (online)
                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│                   MICROSERVICIO (nube)  ── este plan               │
│                                                                    │
│   FastAPI  ──►  asistente.consultar()                              │
│                    ├── clasificador.py   (CNN EfficientNet-B4)     │
│                    ├── nlp_texto.py      (síntomas)                │
│                    ├── fusion.py         (imagen + texto)          │
│                    ├── almacen + busqueda (TF-IDF + BERT)          │
│                    └── generador.py ──►  Ollama / Qwen (grande)    │
│                                                                    │
│   Almacén: SQLite + embeddings (solo lectura en servicio)         │
└──────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────┐
                    │  Ollama (contenedor aparte)│
                    │  modelo: qwen3.5:4b        │
                    └───────────────────────────┘
```

**Idea clave:** el mismo código de `modulos/` se reutiliza en los dos lados. En la
nube se empaqueta como API con modelo grande; en el móvil se empaqueta con modelos
pequeños. El microservicio NO necesita `conexion.py` (en el servidor siempre hay red).

---

## 3. Stack técnico del microservicio

| Componente | Elección | Por qué |
|---|---|---|
| Framework API | **FastAPI** | Async, validación con Pydantic, documentación OpenAPI/Swagger automática |
| Servidor ASGI | **Uvicorn** (+ Gunicorn en prod) | Estándar para FastAPI; workers para concurrencia |
| Modelos ML | PyTorch (CNN) + sentence-transformers (BERT) | Ya en el proyecto |
| LLM | **Ollama** apuntado por `OLLAMA_URL` | `generador.py` ya lo soporta por variable de entorno |
| Contenedores | **Docker** + **docker-compose** | Aísla API y Ollama; reproducible |
| Almacén | SQLite + `embeddings_bert.pkl` (solo lectura) | Sin servidor de BD aparte; el corpus cambia poco |

---

## 4. Diseño de la API

### Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/salud` | Health check (¿modelos cargados? ¿Ollama responde?) |
| `GET` | `/cultivos` | Lista de cultivos con cobertura de documentos |
| `POST` | `/diagnosticar` | **Principal**: imagen + texto + rol + cultivos → diagnóstico |
| `POST` | `/clasificar` | Solo CNN (foto → cultivo/enfermedad). Útil para pruebas |
| `GET` | `/docs` | Swagger UI (lo da FastAPI gratis) |

### Contrato de `/diagnosticar`

**Request** (`multipart/form-data`):
```
imagen   : archivo (jpg/png)        [requerido]
texto    : string  ("" si no hay)   [opcional]
rol      : "agricultor" | "aprendiz"
cultivos : "calabaza,maiz"          [opcional; CSV. Vacío = sin filtro]
```

**Response** (`application/json`):
```json
{
  "modo": "online",
  "diagnostico": {
    "cultivo": "calabaza",
    "enfermedad": "oídio",
    "confianza_original": 0.47,
    "confianza_ajustada": 0.62,
    "explicacion": "El texto refuerza el diagnóstico..."
  },
  "sintomas": ["oidio", "polvo blanco"],
  "avisos": ["La confianza de la imagen es baja (47%)..."],
  "respuesta": {
    "texto": "DIAGNÓSTICO: ... TRATAMIENTO: ...",
    "tratamiento": "...",
    "prevencion": "...",
    "fuentes": ["Guía ... INIFAP 2020"]
  },
  "n_documentos": 9
}
```

> El contrato calza casi 1:1 con lo que ya devuelve `asistente.consultar()`. El
> endpoint es una capa fina sobre esa función.

---

## 5. Estructura de carpetas (lo nuevo)

```
modelo/
├── modulos/                  # YA EXISTE (lógica reutilizada)
├── api/                      # NUEVO — el microservicio
│   ├── main.py               # app FastAPI, endpoints
│   ├── esquemas.py           # modelos Pydantic (request/response)
│   ├── dependencias.py       # carga de modelos al arrancar (singletons)
│   └── config.py             # lectura de variables de entorno
├── Dockerfile                # imagen del microservicio
├── docker-compose.yml        # API + Ollama
├── .dockerignore
└── requirements-api.txt      # FastAPI, uvicorn, gunicorn, python-multipart
```

---

## 6. Carga de modelos y rendimiento

Los modelos son pesados (CNN ~71 MB, BERT ~120 MB). **No** se pueden cargar por petición.

- **Warm-up al arrancar:** cargar CNN y BERT en el evento `startup` de FastAPI (una sola
  vez por proceso). `clasificador.py` y `busqueda_semantica.py` ya usan carga perezosa
  con singleton — solo hay que "tocarlos" al inicio para precargar.
- **Concurrencia:** la inferencia PyTorch es bloqueante. Ejecutarla en un *threadpool*
  (`run_in_threadpool`) para no bloquear el event loop. Limitar workers según CPU/RAM.
- **Gunicorn + Uvicorn workers:** cada worker carga su copia de los modelos → dimensionar
  RAM (≈ 2-3 GB por worker con PyTorch CPU). Empezar con 1-2 workers.
- **Ollama aparte:** el LLM corre en su propio contenedor; varias peticiones lo comparten.

---

## 7. Contenerización

### Dockerfile (esquema)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
# Torch CPU (más liviano que la versión CUDA) salvo que haya GPU
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-api.txt
# Pre-descargar el modelo BERT dentro de la imagen (para arrancar offline-friendly)
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"
COPY . .
EXPOSE 8000
CMD ["gunicorn","api.main:app","-k","uvicorn.workers.UvicornWorker", \
     "-w","2","-b","0.0.0.0:8000","--timeout","180"]
```

### docker-compose.yml (esquema)
```yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      - OLLAMA_URL=http://ollama:11434/api/generate
      - QWEN_MODELO=qwen3.5:4b        # en la nube SÍ cabe el grande
    depends_on: [ollama]
    volumes:
      - ./datos:/app/datos:ro         # corpus en solo lectura

  ollama:
    image: ollama/ollama:latest
    ports: ["11434:11434"]
    volumes: [ollama_models:/root/.ollama]
    # GPU opcional: deploy.resources.reservations.devices (nvidia)

volumes:
  ollama_models:
```

> Tras levantar: `docker compose exec ollama ollama pull qwen3.5:4b` (una vez).

---

## 8. Configuración por entorno

Todo lo sensible/variable, por variables de entorno (ya empezado en `generador.py`):

| Variable | Dev | Producción (nube) |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | `http://ollama:11434/api/generate` |
| `QWEN_MODELO` | `qwen3.5:0.8b` | `qwen3.5:4b` (o mayor) |
| `OLLAMA_TIMEOUT` | `120` | `180` |
| `MAX_IMAGEN_MB` | `8` | `8` |

---

## 9. Seguridad

- **Validar la imagen:** tipo MIME (jpg/png), tamaño máximo (p. ej. 8 MB), dimensiones
  razonables. Rechazar lo demás (evita el caso del PDF/HTML corrupto que ya vimos).
- **Límite de tasa (rate limiting):** por IP/clave, para que la API no se sature.
- **Autenticación:** API key o token (el backend de la app la presenta). El servicio
  NO maneja login de usuarios finales.
- **HTTPS** por delante (reverse proxy: Nginx/Caddy o el del proveedor cloud).
- **Sin secretos en el código:** todo por variables de entorno / gestor de secretos.
- **Regla del dominio se mantiene:** responder solo desde documentos; nunca inventar
  dosis (ya reforzado en `generador.py`).

---

## 10. Observabilidad

- **Logs estructurados:** por petición → cultivo/enfermedad detectados, confianza, modo,
  nº de documentos, latencia, modelo usado. (Sin guardar imágenes por privacidad.)
- **`/salud`:** comprueba que CNN, BERT y Ollama responden. Para readiness/liveness.
- **Métricas (opcional):** latencia p50/p95, peticiones/min, tasa de error → Prometheus.

---

## 11. Escalabilidad y despliegue

- **CPU vs GPU:** CNN + BERT en CPU sirven para carga moderada. El **LLM grande** es lo
  que más se beneficia de GPU. Para escalar, separar: API en CPU, Ollama en GPU.
- **Escalado horizontal:** la API es **stateless** (el corpus es solo lectura; los
  cultivos vienen en la petición) → se pueden levantar N réplicas detrás de un balanceador.
- **Dónde hostear:** cualquier nube con contenedores (Render, Railway, Fly.io, AWS ECS,
  GCP Cloud Run con ajustes, un VPS con Docker). Ollama con GPU encarece; valorar un
  endpoint LLM gestionado si el costo aprieta.

---

## 12. Plan por fases (pasos concretos)

| Fase | Entregable | Detalle |
|---|---|---|
| **M1** | API mínima | FastAPI con `/salud` y `/diagnosticar` llamando a `asistente.consultar()`. Probar local con Uvicorn. |
| **M2** | Carga warm-up | Precargar CNN + BERT en `startup`; inferencia en threadpool. Medir latencia. |
| **M3** | Validación + errores | Validar imagen (tipo/tamaño), manejo de errores (Ollama caído → 503 claro). |
| **M4** | Dockerización | `Dockerfile` + `docker-compose` (API + Ollama). Levantar todo con un comando. |
| **M5** | Config + seguridad | Variables de entorno, API key, rate limiting, límites de tamaño. |
| **M6** | Observabilidad | Logs por petición, `/salud` completa, (opcional) métricas. |
| **M7** | Despliegue | Subir a la nube elegida, HTTPS, prueba end-to-end desde un cliente. |

> Recomendación: M1–M2 primero (API funcionando local), validar contra la interfaz
> Gradio actual apuntándola a la API, y solo entonces contenerizar (M4).

---

## 13. Pruebas

- **Unitarias:** la lógica ya tiene tests en `tests/`. Añadir tests de los endpoints
  con `TestClient` de FastAPI (sin levantar servidor).
- **Generador falso:** reutilizar la inyección `fn_generar` para probar los endpoints
  sin depender de Ollama (igual que en los tests actuales).
- **Carga:** una prueba simple de concurrencia (p. ej. `locust` o `hey`) para dimensionar
  workers y memoria.

---

## 14. Riesgos y decisiones pendientes

- **Tamaño de la imagen Docker:** PyTorch infla la imagen (~2-3 GB). Mitigar con torch CPU,
  multi-stage build y `.dockerignore` (excluir `Entrenamiento/`, PDFs, venv).
- **Costo del LLM grande en la nube:** GPU para Ollama es caro. Decidir: GPU propia vs
  modelo mediano en CPU vs endpoint LLM gestionado.
- **Sincronizar corpus:** cómo se actualiza `datos/` (SQLite + embeddings) en producción.
  Hoy se regenera con `construir_corpus.py`; definir si se hornea en la imagen o se monta
  como volumen.
- **Paridad nube/dispositivo:** asegurar que la versión offline del móvil dé respuestas
  coherentes con la nube (mismos documentos, prompts equivalentes).

---

## Resumen en una línea

Una **API FastAPI** fina sobre `asistente.consultar()`, con CNN+BERT precargados y Qwen
grande vía Ollama, **contenerizada** (API + Ollama), **stateless** y escalable, que
atiende al móvil **cuando hay cobertura** — mientras el modo offline sigue viviendo en el
dispositivo con los modelos pequeños.
