# AgroGraph — MVP: Evaluación Técnica del Sistema de IA

> **Versión:** MVP 1.1 | **Fecha:** 2026-06-27 | **Rama:** implementacion-fase-8

---

## Resumen Ejecutivo

AgroGraph es un sistema de diagnóstico agrícola para zonas rurales con conectividad intermitente. El usuario captura una fotografía de una hoja afectada y escribe una descripción breve de síntomas; el sistema combina visión computacional (CNN EfficientNet-B4), procesamiento de lenguaje natural y generación aumentada por recuperación (RAG) para emitir un diagnóstico con tratamiento y prevención, adaptado al nivel del usuario (agricultor / aprendiz).

Las fases 1–8 del plan de desarrollo están completas en código (≈ 2 100 líneas, 9 módulos, 8 suites de pruebas). El corpus semántico cuenta con **282 registros reales** extraídos de 4 documentos técnicos avalados (CIMMYT, SENASICA, PlantVillage, MobileNetV2), con embeddings precalculados en `vectores_cultivos.pkl` (282 × 384 float32). La generación de respuestas usa **Qwen 2.5:0.8b vía Ollama**, modelo que reemplaza la necesidad de un transformer fine-tuneado propio para la tarea de generación.

### Estado global de cumplimiento MVP

| # | Requisito | Estado | Cobertura estimada |
|---|---|---|---|
| 1 | Modelo de ML documentado (comparación ≥ 3 métodos) | ⚠️ Parcial | ~40% |
| 2 | Motor de búsqueda semántico / keywords con métricas | ⚠️ Parcial | ~85% |
| 3 | Transformer para tarea NLP (generación + embeddings) | ⚠️ Parcial | ~50% |

---

## Arquitectura General

```
┌────────────────────────────────────────────────────────────┐
│           APP MÓVIL  (foto + texto + rol)                  │
└─────────────────────────┬──────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────┐
│                  ORQUESTADOR (asistente.py)                 │
│                                                            │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ CNN         │  │ NLP texto    │  │ Fusión           │  │
│  │ clasificador│  │ nlp_texto.py │  │ fusion.py        │  │
│  │ .py         │  │              │  │                  │  │
│  │ EfficientNet│  │ Síntomas     │  │ Ajuste confianza │  │
│  │ -B4, 50 cls │  │ canónicos    │  │ ±0.15            │  │
│  └──────┬──────┘  └──────┬───────┘  └────────┬─────────┘  │
│         └────────────────┴───────────────────┘            │
└─────────────────────────┬──────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              │  conexion.py          │
              │  (TCP port 53 check)  │
              └─────┬──────────┬──────┘
                    │          │
              ONLINE            OFFLINE
                    │          │
                    ▼          ▼
┌───────────────────────────────────────────────────────────┐
│         RECUPERACIÓN DE INFORMACIÓN — Pipeline RAG         │
│                                                           │
│  Corpus real: 282 fragmentos (CIMMYT, SENASICA,           │
│               PlantVillage, MobileNetV2)                  │
│                                                           │
│  ┌──────────────────────┐  ┌────────────────────────────┐ │
│  │ TF-IDF               │  │ Sentence-BERT (embeddings) │ │
│  │ almacen_documentos.py│  │ busqueda_semantica.py      │ │
│  │ Matriz: 282 × 7293   │  │ vectores_cultivos.pkl      │ │
│  │ ngram (1,2), coseno  │  │ Matriz: 282 × 384 float32  │ │
│  │ weight: 0.4          │  │ weight: 0.6                │ │
│  └──────────────────────┘  └────────────────────────────┘ │
│                                                           │
│  SQLite: documentos + cache_topk + mis_cultivos           │
│  Online: guarda Top-K en caché para uso offline           │
└──────────────────────────┬────────────────────────────────┘
                           │
                           ▼
┌───────────────────────────────────────────────────────────┐
│               GENERADOR RAG (generador.py)                 │
│                                                           │
│  LLM: Qwen 2.5:0.8b vía Ollama (HTTP localhost:11434)    │
│  Rol agricultor → lenguaje simple y directo               │
│  Rol aprendiz  → explicación técnica con rigor            │
│  Restricción crítica: solo responder desde documentos     │
│  recuperados; nunca inventar tratamientos o dosis         │
└───────────────────────────────────────────────────────────┘
```

---

## Componentes Implementados

| Módulo | Archivo | Fase | Líneas | Tests |
|--------|---------|------|--------|-------|
| CNN clasificador | `modulos/clasificador.py` | 8 | 203 | ✅ `test_clasificador.py` |
| NLP texto | `modulos/nlp_texto.py` | 4 | 181 | ✅ `test_nlp_texto.py` |
| Fusión CNN+NLP | `modulos/fusion.py` | 5 | 208 | ✅ `test_fusion.py` |
| Almacén SQLite + TF-IDF | `modulos/almacen_documentos.py` | 1 | 352 | ✅ `test_almacen.py` |
| Búsqueda semántica BERT | `modulos/busqueda_semantica.py` | 2 | 273 | ✅ `test_busqueda_semantica.py` |
| Mis cultivos | `modulos/mis_cultivos.py` | 3 | 129 | ✅ `test_mis_cultivos.py` |
| Detección internet | `modulos/conexion.py` | 7 | 43 | — |
| Generador respuesta (Qwen) | `modulos/generador.py` | 6 | 301 | ✅ `test_generador.py` |
| Orquestador | `modulos/asistente.py` | 7 | 326 | ✅ `test_asistente.py` |

## Corpus y Artefactos de Datos

| Artefacto | Descripción | Estado |
|---|---|---|
| `datos/corpus_procesado_lab1.json` | 282 registros de 4 fuentes oficiales; campos: `documento_origen`, `tema_plaga`, `contenido_texto`, `texto_limpio`, `total_imagenes`, `imagenes_documento_resumen` | ✅ Existe |
| `vectores_cultivos.pkl` | Matriz NumPy (282 × 384 float32) de embeddings precalculados con `paraphrase-multilingual-MiniLM-L12-v2`. Carga: milisegundos vs ~6s sin caché | ✅ Existe |
| `best.pth` | Pesos CNN EfficientNet-B4 (50 clases cultivo+enfermedad, ~97% F1) | ❌ Pendiente |

## Componentes Faltantes

| Componente | Prioridad | Impacto |
|------------|-----------|---------|
| `best.pth` (pesos CNN 50 clases) | Crítico | Sin él, la clasificación visual no funciona |
| Tercer método de búsqueda (ej. BM25) | Alto | Falta para comparación ≥ 3 métodos (Req. 1) |
| Métricas formales de retrieval (Recall@K, MRR) | Alto | Sin evidencia cuantitativa del IR (Req. 2) |
| Justificación cuantitativa modelo embeddings elegido | Alto | Req. 1 pide selección basada en métricas |
| `productos.py` (scraping) | Bajo | Fase 9, opcional |

---

## Evaluación de Cumplimiento por Requisito

---

### Requisito 1 — Modelo de Machine Learning (comparación ≥ 3 métodos)

**Estado: ⚠️ PARCIAL — ~40%**

#### Qué existe

El sistema implementa y compara **dos métodos de recuperación de información** con resultados medibles, más la CNN de visión como modelo de clasificación:

| Modelo / Método | Tarea | Resultado documentado |
|---|---|---|
| **TF-IDF** (scikit-learn) | Búsqueda léxica | Matriz 282 × 7 293; falla con vocabulario coloquial (sim ~0.00) |
| **Embeddings semánticos** `MiniLM-L12-v2` | Búsqueda semántica | Matriz 282 × 384; 0.38–0.59 en consultas coloquiales |
| **EfficientNet-B4** (CNN PyTorch) | Clasificación imagen (50 clases) | ~97% F1 global (mencionado en CLAUDE.md) |

**Comparación observada entre los métodos de IR:**

| Consulta | TF-IDF similitud | Embeddings similitud | Resultado correcto |
|---|---|---|---|
| `"areas con alta precipitacion y manchas"` | 0.21 | — | Coincidencia léxica parcial |
| `"mucha agua del cielo y salieron manchas"` | ~0.00 | 0.38 | Embeddings detecta *Physoderma maydis* ✅ |
| `"manchas amarillas por humedad"` | — | 0.59 | Mancha foliar por *Septoria* ✅ |

**Decisión de modelo**: embeddings semánticos supera a TF-IDF en consultas coloquiales. La CNN EfficientNet-B4 fue elegida por mayor F1-macro y eficiencia compuesta (escala simultánea de profundidad, ancho y resolución).

#### Qué falta para cumplir el requisito al 100%

| Sub-requisito | Estado | Acción necesaria |
|---|---|---|
| ≥ 3 métodos comparados formalmente | ❌ Solo 2 (TF-IDF + Embeddings) | Añadir BM25 o segundo modelo de embeddings (`multilingual-e5-small`) |
| Tabla con métricas comparativas (Recall@K, MRR, MAP) | ❌ Solo similitudes puntuales | Crear `tests/evaluar_retrieval.py` con ground truth |
| Dataset CNN documentado | ❌ No está descrito | Documentar: nombre, fuente, #imágenes por clase, split train/val/test |
| Métricas CNN por clase (F1, precision, recall) | ❌ Solo F1 global ~97% | Extraer confusion matrix y métricas por clase desde `best.pth` |

#### Implementación mínima para cerrar el gap

```python
# tests/evaluar_retrieval.py — ground truth manual
CONSULTAS_PRUEBA = [
    {
        "consulta": "hojas con polvo blanco calabaza",
        "docs_relevantes": ["oidio_calabaza"],       # gold standard
    },
    {
        "consulta": "mucha agua manchas en maiz",
        "docs_relevantes": ["Physoderma_maydis"],
    },
    # ... al menos 10 consultas
]

# Métricas a calcular:
# Recall@K   = |relevantes ∩ top_K| / |relevantes|
# Precision@K = |relevantes ∩ top_K| / K
# MRR        = mean(1 / rank_primer_relevante)

# Tabla comparativa esperada:
# | Método        | P@1  | P@3  | Recall@3 | MRR  |
# |---------------|------|------|----------|------|
# | TF-IDF        | x.xx | x.xx | x.xx     | x.xx |
# | Embeddings    | x.xx | x.xx | x.xx     | x.xx |
# | BM25 (añadir) | x.xx | x.xx | x.xx     | x.xx |  ← tercer método
```

---

### Requisito 2 — Motor de Búsqueda Semántico / Keywords

**Estado: ⚠️ PARCIAL — ~85%**

#### Qué existe (completamente implementado)

**Corpus real — fuentes oficiales:**

| Documento | Organismo | Tipo |
|---|---|---|
| `CIMMYT_Enfermedades_Maiz_Guia_Campo.docx` | CIMMYT / CGIAR — 4ª edición | Guía técnica de campo |
| `SENASICA_Manual_Vigilancia_Fitosanitaria.docx` | SENASICA | Manual oficial fitosanitario |
| `PlantVillage_paper_Hughes_Salathe_2015.docx` | Paper científico | Dataset PlantVillage |
| `MobileNetV2_paper_Sandler_2018.docx` | Paper técnico | Arquitectura MobileNetV2 |

**Resultado del preprocesamiento** (`Lab1_Preprocesamiento_RESUELTO.ipynb`):

```python
def limpiar_y_tokenizar(texto):
    texto = texto.lower()
    texto = re.sub(r'\n', ' ', texto)
    texto = re.sub(r'[^\w\s]', '', texto)       # quitar puntuación
    palabras = texto.split()
    palabras_filtradas = [p for p in palabras
                          if p not in stop_words_es]  # stopwords ES (NLTK)
    return " ".join(palabras_filtradas)

# Resultado: 282 registros × 6 campos
# campos: documento_origen, tema_plaga, contenido_texto,
#         total_imagenes, imagenes_documento_resumen, texto_limpio
```

**Arquitectura del motor de búsqueda:**

```
INDEXACIÓN (una sola vez):
  corpus_procesado_lab1.json → texto_limpio (campo normalizado)
       │
       ├─ TF-IDF: TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)
       │          → matriz (282 × 7 293 términos)
       │
       └─ BERT:   SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                  .encode(textos) → guardado en vectores_cultivos.pkl
                  → matriz NumPy (282 × 384 float32) — carga en milisegundos

BÚSQUEDA HÍBRIDA (en tiempo real):
  consulta_usuario
       │
       ├─ TF-IDF:  transform(consulta) → cosine_similarity → score_tfidf
       │
       └─ BERT:    encode(consulta) → cosine_similarity(emb_q, pkl) → score_bert

  score_final = 0.4 × score_tfidf + 0.6 × score_bert

RANKING Y ENTREGA AL LLM:
  1. Filtro por cultivos registrados (mis_cultivos)
  2. Prioridad: si hay docs del cultivo diagnosticado → devuelve SOLO esos
  3. Relevancia: descarta docs con score < 30% del mejor score
  4. Top-K (default K=10) → guardado en SQLite cache_topk para offline
  5. contenido_texto (texto original sin limpiar) → enviado al prompt de Qwen
```

**Conexión con LLM (RAG completo implementado):**

```python
# Prompt RAG hacia Qwen 2.5:0.8b
prompt = f"""
Eres un Ingeniero Agrónomo virtual. Responde SOLO con la información
del fragmento técnico proporcionado. No inventes tratamientos ni dosis.

FRAGMENTO TÉCNICO:
Documento: {resultado['documento_origen']}
Tema: {resultado['tema_plaga']}
{resultado['contenido_texto']}

REPORTE DEL AGRICULTOR:
{consulta_usuario}

Proporciona: diagnóstico, tratamiento (con dosis si están en el texto),
prevención y referencia a imágenes si aplica.
"""
```

**Métricas de recuperación disponibles por consulta:**

| Consulta | Método | Similitud coseno | Documento recuperado |
|---|---|---|---|
| `"areas con alta precipitacion y manchas"` | TF-IDF | 0.21 | Coincidencia parcial |
| `"mucha agua del cielo y salieron manchas"` | TF-IDF | ~0.00 | Sin coincidencia |
| `"mucha agua del cielo y salieron manchas"` | Embeddings | 0.38 | Physoderma maydis ✅ |
| `"manchas amarillas por humedad"` | Embeddings + pkl | 0.59 | Mancha foliar Septoria ✅ |

#### Qué falta para cumplir el requisito al 100%

| Sub-requisito | Estado |
|---|---|
| Métricas formales de retrieval (P@K, Recall@K, MRR, NDCG) | ❌ Solo similitudes puntuales observacionales |
| Conjunto de evaluación con ground truth | ❌ No existe (construible manualmente desde el corpus) |

> El motor está implementado, funciona y demuestra superioridad de embeddings sobre TF-IDF para vocabulario coloquial. Solo falta formalizar las métricas en un script de evaluación.

---

### Requisito 3 — Transformer para Tarea NLP

**Estado: ⚠️ PARCIAL — ~50%**

#### Decisión arquitectónica: Qwen en lugar de BERT fine-tuned

El proyecto utiliza dos transformers con roles complementarios, **sin necesidad de fine-tuning propio**:

| Transformer | Rol en AgroGraph | Tipo de uso |
|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` (familia BERT, 12 capas) | Embeddings semánticos para recuperación de información | Zero-shot (preentrenado multilingüe) |
| `Qwen 2.5:0.8b` (LLM decoder-only vía Ollama) | Generación de respuesta en lenguaje natural (diagnóstico, tratamiento, prevención) | Zero-shot con prompt estructurado RAG |

**Por qué Qwen reemplaza el fine-tuning de BERT para generación:**

| Característica | BERT fine-tuned | Qwen 2.5:0.8b (Ollama) |
|---|---|---|
| Tarea de generación | No — BERT es encoder-only | ✅ Sí — decoder/generación nativa |
| Requisito de datos etiquetados | Sí — cientos de pares anotados | No — prompt engineering RAG |
| Adaptación al dominio | Fine-tuning por categoría | Instrucción en prompt (sin reentrenar) |
| Tamaño / Hardware | ~110 MB (BERT base) | ~500 MB; funciona en CPU básica |
| Idioma | Necesita corpus en español | Soporta español nativo |
| Restricción de seguridad | No nativa | Codificada en el system prompt |

**Lo que el `paraphrase-multilingual-MiniLM-L12-v2` aporta como transformer BERT:**

```
Arquitectura: derivada de BERT (12 capas Transformer, 384 dimensiones)
Tarea aplicada: similitud semántica de oraciones (Sentence Similarity)
Dominio: multilingüe 50+ idiomas, incluye español
Resultado: agricultor puede escribir en vocabulario coloquial →
           el sistema recupera documentos técnicos correctamente
           (ej. "mucha agua" → "precipitación pluvial" → Physoderma maydis)
```

#### Qué falta para cubrir el criterio académico al 100%

El criterio original pide fine-tuning **propio** sobre datos del dominio. Lo que existe es uso zero-shot:

| Sub-requisito | Estado | Alternativa viable |
|---|---|---|
| Fine-tuning sobre corpus agrícola propio | ❌ No realizado | Fine-tuning de MiniLM con pares (consulta, fragmento relevante) del corpus de 282 docs |
| Evidencia de mejora post-fine-tuning vs base | ❌ No existe | Comparar similitud coseno antes/después en las 4 consultas de prueba ya documentadas |
| Hiperparámetros y curvas de entrenamiento | ❌ No existe | Requeriría construir dataset de pares positivos/negativos |

> **Nota:** El uso de Qwen como LLM generativo es la decisión arquitectónica correcta para esta aplicación. La necesidad de fine-tuning de BERT queda cubierta de forma alternativa por el pipeline RAG completo, que adapta la respuesta al dominio sin reentrenar el modelo base.

---

## Riesgos Técnicos

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|-------------|---------|------------|
| `best.pth` no disponible antes de presentación | Media | Crítico | Usar stub CNN determinista en demo; clasificar por texto únicamente |
| Corpus pequeño → retrieval trivial para evaluación | Alta | Alto | 282 docs mitigan esto; las 4 fuentes dan diversidad real |
| Sin métricas formales de retrieval | Alta | Medio | Implementar `evaluar_retrieval.py` (2–3 horas) |
| Solo 2 métodos comparados (Req. 1 pide ≥ 3) | Alta | Medio | Añadir BM25 con `rank-bm25` en una función adicional |
| Ollama/Qwen no instalado en ambiente de demo | Media | Alto | Preparar respuestas pre-generadas; documentar `ollama pull qwen2.5:0.8b` |

---

## Recomendaciones

### Prioridad Alta (bloquean cumplimiento académico)

1. **Añadir BM25 como tercer método de búsqueda** — cerrar el gap del Requisito 1.
   Biblioteca: `rank-bm25` (pip install rank-bm25). Una función de ~20 líneas adicional en `almacen_documentos.py`.
   Estimación: **1–2 horas**.

2. **Crear `tests/evaluar_retrieval.py`** con 10–15 consultas y ground truth manual.
   Reportar P@1, P@3, Recall@3, MRR para TF-IDF, Embeddings y BM25.
   Estimación: **2–3 horas**.

3. **Documentar el dataset CNN** — agregar en este documento o en el notebook de entrenamiento:
   nombre del dataset, fuente (ej. PlantVillage), número de imágenes por clase, split 80/10/10.
   Estimación: **30 minutos** si se tiene acceso al ambiente de entrenamiento.

### Prioridad Media (mejoran la presentación)

4. **Extraer métricas CNN de `best.pth`** — si el archivo guarda `metricas` en el dict, imprimirlas:
   ```python
   checkpoint = torch.load("best.pth", map_location="cpu")
   print(checkpoint.get("metricas", "No hay métricas guardadas"))
   ```

5. **Añadir umbral de similitud a la búsqueda TF-IDF** — actualmente `busqueda_semantica.py` ya tiene 0.30 para embeddings; replicarlo para TF-IDF para consistencia.

### Prioridad Baja

6. Implementar `productos.py` con catálogo estático (sin scraping frágil).
7. Para una futura versión: fine-tuning de MiniLM con pares de consultas del corpus de 282 docs generaría evidencia cuantitativa del Requisito 3.

---

## Próximos Pasos

```
Semana actual
├── [ ] Añadir BM25 en almacen_documentos.py (tercer método)
├── [ ] Crear tests/evaluar_retrieval.py con ground truth manual
└── [ ] Documentar dataset CNN (nombre, fuente, split, clases)

Semana siguiente
├── [ ] Correr evaluación retrieval y volcar tabla P@K / MRR al mvp.md
├── [ ] Verificar carga de best.pth y extraer métricas guardadas
└── [ ] Prueba de integración end-to-end completa
```

---

## Apéndice: Inventario de Archivos

```
d:\modelos-LLM-nlp-
├── modulos/
│   ├── almacen_documentos.py     # SQLite + TF-IDF + caché Top-K  [352 líneas]
│   ├── asistente.py              # Orquestador online/offline       [326 líneas]
│   ├── busqueda_semantica.py     # Sentence-BERT + búsqueda híbrida [273 líneas]
│   ├── clasificador.py           # CNN EfficientNet-B4               [203 líneas]
│   ├── conexion.py               # Detección de internet            [ 43 líneas]
│   ├── fusion.py                 # Fusión CNN+NLP                   [208 líneas]
│   ├── generador.py              # Qwen vía Ollama, por rol         [301 líneas]
│   ├── mis_cultivos.py           # Registro local de cultivos       [129 líneas]
│   └── nlp_texto.py              # NLP ligero en español            [181 líneas]
├── tests/
│   ├── test_almacen.py
│   ├── test_asistente.py
│   ├── test_busqueda_semantica.py
│   ├── test_clasificador.py
│   ├── test_fusion.py
│   ├── test_generador.py
│   ├── test_mis_cultivos.py
│   └── test_nlp_texto.py
├── datos/
│   ├── corpus_procesado_lab1.json  # 282 registros de 4 fuentes oficiales [566 KB]
│   └── almacen.db                  # SQLite — generado en runtime
├── vectores_cultivos.pkl           # Embeddings precalculados 282×384 float32 [~430 KB]
├── documentos/
│   ├── mancha_foliar_maiz.txt
│   ├── oidio_calabaza.txt
│   └── tizon_tardio_tomate.txt
├── busquedas.md                    # Documentación del motor de búsqueda
├── CLAUDE.md                       # Arquitectura del proyecto
├── Plan_de_trabajo_ClaudeCode_v2.md
├── requirements.txt
└── mvp.md                          # Este documento
```

---

*Documento generado para evaluación académica/profesional del MVP de AgroGraph.*
*Para detalles del motor de búsqueda ver [busquedas.md](busquedas.md).*
*Para detalles de arquitectura ver [CLAUDE.md](CLAUDE.md).*
