# Sistema de Búsqueda Semántica sobre Corpus Agrícola

## Índice

1. [Origen y preprocesamiento del corpus](#1-origen-y-preprocesamiento-del-corpus)
2. [Estructura de `corpus_procesado_lab1.json`](#2-estructura-del-corpus)
3. [Para qué sirve `vectores_cultivos.pkl`](#3-vectores_cultivospkl)
4. [Métodos de búsqueda implementados](#4-métodos-de-búsqueda)
5. [Evaluación contra los criterios del proyecto](#5-evaluación-contra-los-criterios-del-proyecto)
6. [Pipeline RAG y compatibilidad con Ollama Qwen](#6-pipeline-rag-y-compatibilidad-con-ollama-qwen)
7. [Diagrama del flujo completo](#7-diagrama-del-flujo-completo)

---

## 1. Origen y preprocesamiento del corpus

El corpus parte de cuatro documentos `.docx` técnicos avalados por organismos reconocidos:

| Archivo | Fuente |
|---|---|
| `CIMMYT_Enfermedades_Maiz_Guia_Campo.docx` | CIMMYT / CGIAR — guía de campo, 4ª edición |
| `SENASICA_Manual_Vigilancia_Fitosanitaria.docx` | SENASICA — manual oficial fitosanitario |
| `PlantVillage_paper_Hughes_Salathe_2015.docx` | Paper científico dataset PlantVillage |
| `MobileNetV2_paper_Sandler_2018.docx` | Paper arquitectura MobileNetV2 |

El preprocesamiento se realiza en `Lab1_Preprocesamiento_RESUELTO.ipynb` (celda 1) con la función `limpiar_y_tokenizar()`:

```python
def limpiar_y_tokenizar(texto):
    texto = texto.lower()                          # minúsculas
    texto = re.sub(r'\n', ' ', texto)              # quitar saltos de línea
    texto = re.sub(r'[^\w\s]', '', texto)          # quitar puntuación
    palabras = texto.split()
    palabras_filtradas = [p for p in palabras
                          if p not in stop_words_es]   # eliminar stopwords ES (NLTK)
    return " ".join(palabras_filtradas)

df_final_lab1["texto_limpio"] = df_final_lab1["contenido_texto"].apply(limpiar_y_tokenizar)
df_final_lab1.to_json("corpus_procesado_lab1.json", orient='records', force_ascii=False, indent=4)
```

El campo `texto_limpio` es el que alimenta los vectorizadores. El campo `contenido_texto` (texto original) es el que se entrega al LLM como contexto en el paso de generación.

---

## 2. Estructura del corpus

Cada entrada en `corpus_procesado_lab1.json` tiene los siguientes campos:

| Campo | Tipo | Descripción |
|---|---|---|
| `documento_origen` | string | Nombre del `.docx` de origen |
| `tema_plaga` | string | Título del bloque (nombre de plaga o sección del manual) |
| `contenido_texto` | string | Texto original sin modificar — va al LLM |
| `total_imagenes` | int | Total de imágenes del documento de origen |
| `imagenes_documento_resumen` | string | Lista de nombres de archivos de imagen |
| `texto_limpio` | string | Texto normalizado sin stopwords — va a los vectorizadores |

**Dimensiones:** 282 registros × 6 campos.

---

## 3. `vectores_cultivos.pkl`

Archivo binario serializado con `pickle` que contiene una **matriz NumPy de embeddings precalculados**:

```
Forma:   (282, 384)   — un vector por documento, 384 dimensiones
Tipo:    float32
Tamaño:  ~430 KB
```

**Por qué existe:** sin este archivo, cada arranque del buscador requiere codificar los 282 textos del corpus a través del modelo (~6 segundos en CPU). Con él, la carga es instantánea y solo se vectoriza la consulta entrante.

**Cómo se generó:**

```python
modelo_embeddings = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
vectores_corpus = modelo_embeddings.encode(df_corpus["texto_limpio"].tolist(), show_progress_bar=True)

with open("vectores_cultivos.pkl", "wb") as f:
    pickle.dump(vectores_corpus, f)
```

**Por qué este modelo:** `paraphrase-multilingual-MiniLM-L12-v2` soporta español de forma nativa (entrenado en 50+ idiomas), está optimizado para similitud semántica de frases, pesa ~117 MB y corre en CPU. Permite que el agricultor describa síntomas con vocabulario coloquial y el sistema entienda el concepto aunque las palabras exactas no estén en el manual.

---

## 4. Métodos de búsqueda

### Método 1 — TF-IDF + Similitud de Coseno

```python
vectorizador = TfidfVectorizer()
matriz_tfidf = vectorizador.fit_transform(df_corpus["texto_limpio"])
# Resultado: matriz (282, 7293) — 282 docs × 7293 términos únicos

def buscar_diagnostico(consulta, top_n=3):
    vector_consulta = vectorizador.transform([consulta.lower()])
    similitudes = cosine_similarity(vector_consulta, matriz_tfidf).flatten()
    indices_top = similitudes.argsort()[-top_n:][::-1]
```

Representa cada documento como un vector de pesos TF-IDF (relevancia del término en el documento vs. rareza en el corpus). La similitud de coseno mide el ángulo entre vectores: 1.0 = idéntico, 0.0 = sin relación.

**Limitación:** exige coincidencia léxica. "Mucha agua del cielo" no conecta con "precipitación pluvial abundante".

### Método 2 — Embeddings Semánticos + Similitud de Coseno

```python
vector_consulta = modelo_embeddings.encode([consulta_usuario])
similitudes = cosine_similarity(vector_consulta, vectores_corpus).flatten()
idx_mejor = similitudes.argmax()
# Umbral mínimo de confianza: 0.30
```

Convierte tanto documentos como consulta en vectores densos de 384 dimensiones que capturan **significado semántico**, no palabras exactas.

### Comparación de resultados observados

| Consulta | Método | Similitud | Resultado |
|---|---|---|---|
| `"areas con alta precipitacion y manchas"` | TF-IDF | 0.21 | Coincidencia léxica parcial |
| `"mucha agua del cielo y salieron manchas"` | TF-IDF | ~0.00 | Sin coincidencia (vocabulario diferente) |
| `"mucha agua del cielo y salieron manchas"` | Embeddings | 0.38 | Detecta Physoderma maydis correctamente |
| `"manchas amarillas por humedad"` | Embeddings + PKL | 0.59 | Mancha foliar por Septoria (carga instantánea) |

---

## 5. Evaluación contra los criterios del proyecto

### Criterio 1 — Comparar al menos 3 modelos y decidir basado en métricas

**Cobertura: parcial (~40%)**

| Lo que está implementado | Lo que falta |
|---|---|
| 2 enfoques comparados: TF-IDF vs Embeddings semánticos | Un tercer enfoque (ej. BM25 o un segundo modelo de embeddings como `multilingual-e5-small`) |
| Métricas observacionales por consulta (similitud de coseno) | Tabla formal de evaluación: MRR, Precision@k o MAP sobre un conjunto de consultas de prueba |
| Decisión implícita: embeddings supera a TF-IDF en consultas coloquiales | Justificación cuantitativa que soporte la elección del modelo final |

Lo que sí se puede argumentar: el notebook documenta que TF-IDF falla con vocabulario diferente (similitud ~0) mientras que el modelo de embeddings alcanza 0.38–0.59 en las mismas consultas, lo cual constituye una comparación funcional entre dos estrategias distintas.

---

### Criterio 2 — Motor de búsqueda por Keywords (TF-IDF/BM25) o Embeddings con métricas

**Cobertura: cumple (~85%)**

Ambos motores están implementados y funcionando:

| Motor | Implementación | Métricas disponibles |
|---|---|---|
| **TF-IDF** (keywords) | `TfidfVectorizer` de scikit-learn | Vocabulario: 7,293 términos; Matriz: (282 × 7,293); Similitud de coseno por consulta |
| **Embeddings semánticos** | `paraphrase-multilingual-MiniLM-L12-v2` | Matriz: (282 × 384); Similitud de coseno por consulta; Umbral de confianza: 0.30 |

Lo que podría sumarse para completar el criterio: calcular métricas formales de recuperación (P@1, P@3, MRR) sobre un pequeño conjunto de preguntas con respuesta esperada conocida, que ya es posible construir manualmente desde el corpus existente.

---

### Criterio 3 — Transformer tipo BERT fine-tuneado para una tarea NLP

**Cobertura: parcial (~30%)**

| Lo que está presente | Lo que falta |
|---|---|
| Se usa `paraphrase-multilingual-MiniLM-L12-v2`, que es una variante de Sentence-BERT (arquitectura derivada de BERT, 12 capas, multilingüe) | Fine-tuning propio sobre datos del dominio agrícola |
| El modelo se aplica a la tarea de **recuperación semántica de información** (búsqueda por similitud de frases) | Entrenamiento supervisado con pares (consulta, fragmento relevante) del corpus de cultivos |
| Corre de forma efectiva en el dominio agrícola en español sin ajuste | Evidencia de que el fine-tuning mejora el score respecto al modelo base |

En resumen: se usa un transformer preentrenado de la familia BERT para la tarea correcta (similitud semántica), pero aplicado de forma **zero-shot** (sin fine-tuning propio). El criterio quedaría cubierto si se añade una etapa de ajuste fino con pares de consultas anotadas del propio corpus.

---

### Resumen de cobertura

| Criterio | Estado | Porcentaje estimado |
|---|---|---|
| Comparar ≥ 3 modelos con métricas y elegir uno | Parcial — solo 2 métodos comparados, métricas observacionales | ~40% |
| Motor TF-IDF o Embeddings con métricas | Cumple — ambos implementados con similitud de coseno reportada | ~85% |
| Transformer tipo BERT fine-tuneado | Parcial — se usa Sentence-BERT en modo zero-shot, sin fine-tuning propio | ~30% |

---

## 6. Pipeline RAG y compatibilidad con Ollama Qwen

La celda 5 del notebook implementa el patrón **RAG** (Retrieval-Augmented Generation) completo. El buscador de embeddings recupera el fragmento más relevante del corpus y lo inyecta en un prompt estructurado que se envía al LLM:

**Estructura del prompt:**
1. **Instrucción de sistema** — rol de Ingeniero Agrónomo virtual + regla crítica de no inventar tratamientos fuera del manual.
2. **Contexto técnico** — fragmento exacto del manual recuperado (documento origen + plaga identificada + texto oficial).
3. **Reporte del agricultor** — consulta tal como la escribió el usuario.
4. **Plantilla de respuesta** — formato solicitado: saludo, explicación simple, acciones, referencias a imágenes.

**Integración con `ollama qwen2.5:0.8b`:**

```python
import requests

def llamar_ollama(prompt):
    respuesta = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:0.8b",
            "prompt": prompt,
            "stream": False
        }
    )
    return respuesta.json()["response"]

# Reemplazar print(prompt_final) en simular_pipeline_sugerencia() por:
respuesta_llm = llamar_ollama(prompt_final)
print(respuesta_llm)
```

**Por qué el corpus es apto para Qwen 0.8B:**

| Característica | Estado |
|---|---|
| Fuente avalada | CIMMYT (CGIAR) y SENASICA — organismos internacionales reconocidos |
| Idioma | Español, compatible con el tokenizador de Qwen |
| Tamaño del contexto | Cada fragmento ~300 chars — entra sin problema en la ventana de Qwen 0.8B |
| Sin alucinaciones de datos | La instrucción del prompt prohíbe al LLM inventar información fuera del fragmento |
| Recursos visuales | `imagenes_documento_resumen` permite citar imágenes sin enviarlas al LLM |
| Hardware | `qwen2.5:0.8b` (~500 MB) + `MiniLM-L12-v2` (~117 MB) — funcional en CPU básica |

**Notas para producción:**
- Regenerar `vectores_cultivos.pkl` cada vez que se agreguen documentos al corpus.
- El umbral 0.30 es ajustable: subirlo reduce falsos positivos, bajarlo amplía cobertura.
- Para respuestas más elaboradas, `qwen2.5:3b` o `qwen2.5:7b` no requieren cambios en el pipeline.

---

## 7. Diagrama del flujo completo

```
documentos/*.docx (CIMMYT, SENASICA, PlantVillage, MobileNetV2)
        │
        │  Lab1_Preprocesamiento.ipynb — celda 1
        │  - lowercase, quitar puntuación, stopwords ES (NLTK)
        ▼
corpus_procesado_lab1.json
(282 registros: tema_plaga, contenido_texto, texto_limpio, imagenes_*)
        │
        │  SentenceTransformer.encode()  ──── una sola vez ────┐
        │  paraphrase-multilingual-MiniLM-L12-v2               │
        ▼                                                       ▼
  [Motor TF-IDF]                                   vectores_cultivos.pkl
  matriz (282 × 7293)                              matriz NumPy (282 × 384)
        │                                                       │
        └───────────────────┬───────────────────────────────────┘
                            │
                   CONSULTA DEL USUARIO
                   "síntomas de la planta..."
                            │
              ┌─────────────┴──────────────┐
              │ TF-IDF cosine similarity   │ Embedding cosine similarity
              │ (búsqueda por palabras)    │ (búsqueda por significado)
              └─────────────┬──────────────┘
                            │
                  Fragmento relevante del corpus
                  (score > 0.30 para embeddings)
                            │
                  Construcción del Prompt RAG
                  (contexto + reglas + plantilla)
                            │
                  Ollama — qwen2.5:0.8b
                            │
                  Respuesta en lenguaje natural
                  para el agricultor
```
