# Sistema de Diagnóstico Agrícola — Backend IA (CNN · NLP · RAG · LLM · Clustering)

Diagnóstico de enfermedades en cultivos a partir de **una foto + un texto de síntomas**,
para zonas con **poca o nula cobertura**. Backend de IA en Python con microservicio REST
(FastAPI), que sirve a la app móvil **AgroGraph-MAS** (Flutter, repositorio aparte).

> 📘 **Documentación técnica completa:** [`docs/DOCUMENTACION_TECNICA.md`](docs/DOCUMENTACION_TECNICA.md)
> (arquitectura, módulos, endpoints, decisiones, integración, diagramas).
> Contexto y decisiones del proyecto: [`CLAUDE.md`](CLAUDE.md).

---

## ¿Qué hace?

```
Imagen → CNN (cultivo+enfermedad) ─┐
Texto  → NLP (síntomas) ───────────┤→ Fusión → Recuperación (TF-IDF+BERT) → LLM (Qwen) → Diagnóstico
```

Regla central de seguridad: las respuestas salen **solo de documentos**; el sistema
**nunca inventa** dosis ni productos.

## Estructura

```
modelo/
├── modulos/     NÚCLEO de dominio (clasificador, nlp_texto, fusion, almacen_documentos,
│                busqueda_semantica, mis_cultivos, conexion, generador, asistente, clustering)
├── app/         MICROSERVICIO REST (FastAPI): main, config, schemas, servicio, db, offline, campanias
├── scripts/     Construcción de corpus y evaluación/entrenamiento (no producción)
├── tests/       Pruebas por módulo
├── datos/       SQLite, índices, corpus, campanias/*.csv
├── documentos/  Fuentes curadas (.txt)
├── modelos/     Pesos (best.pth, modelo_beto/, clustering_*.pkl — no versionados)
├── docs/        Documentación y reportes de métricas
├── ejecutar.py  CLI · interfaz.py  UI de pruebas (Gradio)
└── requirements.txt · CLAUDE.md
```

**Arquitectura:** `modulos/` es el núcleo reutilizable; `app/`, `ejecutar.py`, `interfaz.py`
y `scripts/` solo lo *invocan*. Ver [documentación técnica](docs/DOCUMENTACION_TECNICA.md).

## Puesta en marcha

```powershell
pip install -r requirements.txt
set PYTHONUTF8=1
ollama pull qwen3.5:0.8b                # LLM (modo online)
python scripts/construir_corpus.py      # corpus + índices (una vez)
```

**Uso por consola:**
```powershell
python ejecutar.py --imagen "ruta/foto.jpg" --texto "polvo blanco en hojas" --rol agricultor
```

**Interfaz web de pruebas:**
```powershell
python interfaz.py        # http://localhost:7860
```

## Microservicio REST (modo online)

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000    # Swagger en /docs
```

| Método | Ruta | Descripción |
|---|---|---|
| POST | `/api/v1/consultar` | Diagnóstico (CNN result + texto → RAG + LLM) |
| GET | `/api/v1/inferences` · `/{id}` | Historial de inferencias |
| POST | `/api/v1/clustering/inferir` | Clustering fitosanitario (no supervisado) |
| GET | `/api/v1/clustering/mapa` · `/mapa-campanias` | Mapa epidemiológico (diagnósticos / campañas reales) |
| GET | `/api/v1/offline/catalog` · `/documents/{id}` | Corpus + embeddings para RAG on-device |
| GET | `/health` · `/ready` | Salud / readiness |

Detalle de contratos: [documentación técnica §4 y §13](docs/DOCUMENTACION_TECNICA.md#4-rutas-endpoints-rest).

## Modelos y métricas

| Módulo | Métrica | Reporte |
|---|---|---|
| CNN EfficientNet-B4 (50 clases) | ~0.97 F1 | [comparación](docs/METRICAS_cnn_comparacion.md) |
| Motor de búsqueda (TF-IDF/BERT) | MRR 0.747 | [métricas](docs/METRICAS_busqueda.md) |
| BETO fine-tuneado (texto→cultivo) | F1 0.868 | [métricas](docs/METRICAS_beto.md) |
| Clustering campañas (real) | silhouette 0.359 | [métricas](docs/METRICAS_clustering_campanias.md) |

Reproducir:
```powershell
python scripts/evaluar_busqueda.py
python scripts/finetune_beto.py
python scripts/comparar_cnns.py
python scripts/entrenar_clustering_campanias.py
```

---

*Universidad Politécnica de Chiapas — proyecto de Minería de datos / LLM-NLP.*
