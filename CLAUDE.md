# CLAUDE.md — Contexto del proyecto

> Pega este archivo en la raíz del proyecto. Claude Code lo lee al arrancar.
> Ajusta lo que esté entre [corchetes].

## Qué es este proyecto

Sistema de diagnóstico agrícola para **zonas con poca o nula cobertura** (el campo). El usuario toma una foto de una hoja y escribe un texto breve de síntomas; el sistema:
1. Detecta cultivo + enfermedad (CNN ya entrenada).
2. Interpreta el texto (NLP) y lo fusiona con la imagen.
3. Recupera documentos relevantes (IR: TF-IDF + BERT) y genera con un LLM el diagnóstico, tratamiento, prevención y productos.

Funciona **online/offline con degradación**:
- **Online:** busca fresco, guarda el TOP-K en SQLite (caché).
- **Offline:** responde con el caché local (SQLite + TF-IDF + BERT).

## Reglas de seguridad del dominio (IMPORTANTES)

- Da consejos agrícolas que pueden causar daño si son erróneos. Las respuestas deben salir **solo de los documentos**; nunca inventar tratamientos, dosis ni productos.
- Si la información no está en los documentos, decir claramente que no la tiene.
- Respuestas orientativas; recomendar confirmar con un agrónomo.

## Decisiones de arquitectura fijadas

- **Una sola CNN** (EfficientNet-B4, 50 clases, ya entrenada, ~97% F1). NO 15 modelos. Predice cultivo+enfermedad junto (ej. `Tomato___Late_blight`). El class_mapping viene dentro de best.pth.
- **Búsqueda híbrida TF-IDF + BERT** (Sentence-BERT, modelo multilingüe `paraphrase-multilingual-MiniLM-L12-v2`, local/offline).
- **Caché en SQLite por TOP-K**: se guardan los 10 mejores documentos por enfermedad. Se pre-carga ("semilla") según los cultivos del usuario.
- **"Mis cultivos"**: función local (tabla SQLite) donde el usuario registra los cultivos de su parcela. Filtra búsquedas y pre-carga. NO es sistema de cuentas.
- **3 roles**: agricultor (respuesta simple), aprendiz (respuesta técnica), gestor (admin; exploratorio, ver Fase 10). El login se maneja en el back/app móvil, fuera del código Python del RAG.

## Stack técnico

- **Lenguaje:** Python [3.11]
- **LLM:** Qwen vía **Ollama** (API HTTP local en http://localhost:11434). [modelo: qwen3:4b — evaluar uno más pequeño para móvil]
- **Embeddings/búsqueda semántica:** **BERT** vía Sentence-BERT (sentence-transformers).
- **Búsqueda léxica:** TF-IDF (scikit-learn).
- **Base local:** SQLite (documentos, caché Top-K, mis cultivos).
- **CNN:** EfficientNet-B4 (PyTorch), pesos en `best.pth`.

## Módulos

- `clasificador.py` — CNN: foto → cultivo+enfermedad. (ya existe best.pth)
- `nlp_texto.py` — interpreta el texto del usuario.
- `fusion.py` — combina CNN + NLP.
- `mis_cultivos.py` — cultivos de la parcela (SQLite).
- `almacen_documentos.py` — SQLite + TF-IDF + caché Top-K.
- `busqueda_semantica.py` — Sentence-BERT + búsqueda híbrida.
- `conexion.py` — detecta internet.
- `generador.py` — Qwen redacta según rol.
- `productos.py` — catálogo (scraping). [opcional]
- `asistente.py` — orquestador (online/offline, caché).

## Convenciones

- Código y comentarios **en español**.
- Funciones pequeñas con docstrings claros.
- Cada módulo con prueba en `tests/` o bloque de prueba.
- Manejo de errores explícito (HTTP a Ollama, lecturas de archivos, sin conexión).
- Dependencias mínimas; preferir lo más simple que funcione.

## Estado actual

- [x] CNN entrenada (best.pth, 50 clases, ~97%).
- [ ] Resto de fases (ver Plan_de_trabajo_ClaudeCode_v2.md).

## Decisiones pendientes

- Dónde corren Qwen y BERT en modo offline (móvil con modelos pequeños vs laptop local como servidor de zona). Afecta el empaquetado final.
- Fuente de documentos: locales por ahora; Google Drive en la versión final.
- Fuente para scraping de productos: [por definir].
- Requisito de Minería: posiblemente cubierto por el mapa epidemiológico (Fase 10). Confirmar con el profesor.
- Gestor/mapa epidemiológico: idea aún en exploración; no construir hasta que el núcleo funcione.
