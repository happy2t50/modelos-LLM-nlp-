# Sistema de Diagnóstico Agrícola (CNN + NLP + RAG + LLM)

Diagnóstico de enfermedades en cultivos a partir de **una foto + un texto de síntomas**,
pensado para zonas con **poca o nula cobertura**. Funciona online (nube) y offline
(dispositivo). Ver contexto completo en [CLAUDE.md](CLAUDE.md).

## Estructura del proyecto

```
modelo/
├── modulos/        ← NÚCLEO: la lógica del sistema (lo que corre en producción)
│   ├── clasificador.py        CNN EfficientNet-B4 (foto → cultivo+enfermedad)
│   ├── nlp_texto.py           interpreta el texto del usuario
│   ├── fusion.py              combina imagen + texto
│   ├── almacen_documentos.py  SQLite + TF-IDF + caché Top-K
│   ├── busqueda_semantica.py  BERT + búsqueda híbrida
│   ├── mis_cultivos.py        cultivos de la parcela
│   ├── conexion.py            detección de internet
│   ├── generador.py           Qwen (Ollama) redacta según rol
│   └── asistente.py           orquestador (une todo)
│
├── ejecutar.py     ← punto de entrada por consola (CLI)
├── interfaz.py     ← punto de entrada web de pruebas (Gradio)
│
├── scripts/        ← construcción y evaluación (NO corren en producción)
│   ├── construir_corpus.py    arma el corpus combinado y los índices
│   ├── evaluar_busqueda.py    métricas del motor de búsqueda (Req. #2)
│   ├── finetune_beto.py       fine-tuning de BETO (Req. #3)
│   └── comparar_cnns.py       comparación de 3 CNNs (Req. #1)
│
├── modelos/        ← pesos de modelos (no versionados; pesados)
│   ├── best.pth               CNN entrenada (50 clases, ~97% F1)
│   └── modelo_beto/           BETO fine-tuneado (se regenera)
│
├── datos/          ← base SQLite, índices y corpus (se regeneran)
├── documentos/     ← documentos fuente curados (.txt)
├── tests/          ← pruebas de cada módulo
├── docs/           ← documentación y reportes de métricas
│   ├── Plan_de_trabajo_ClaudeCode_v2.md
│   ├── PLAN_microservicio.md
│   ├── METRICAS_busqueda.md / METRICAS_beto.md / METRICAS_cnn_comparacion.md
│   ├── mvp.md / busquedas.md   (documentación del equipo)
├── extras/         ← material no esencial (no usado por el código)
├── requirements.txt
└── CLAUDE.md
```

**Idea clave de la arquitectura:** `modulos/` es el corazón reutilizable; los puntos de
entrada (`ejecutar.py`, `interfaz.py`) y los `scripts/` solo lo *invocan*. Eso permite
empaquetar el mismo núcleo como microservicio en la nube o embebido en el móvil.

## Puesta en marcha

```powershell
# 1) Dependencias
pip install -r requirements.txt

# 2) Acentos en consola (Windows)
set PYTHONUTF8=1

# 3) Construir el corpus + índices (una vez, o al cambiar documentos)
python scripts/construir_corpus.py

# 4a) Usar por consola
python ejecutar.py --imagen "ruta/foto.jpg" --texto "polvo blanco en hojas" --rol agricultor

# 4b) O usar la interfaz web de pruebas
python interfaz.py        # abre http://localhost:7860
```

> El modo **online** requiere [Ollama](https://ollama.com) corriendo con el modelo:
> `ollama pull qwen3.5:0.8b`. El modo **offline** responde desde el caché local.

## Reproducir las métricas (requisitos del curso)

```powershell
python scripts/evaluar_busqueda.py     # Req. #2 — métricas del motor de búsqueda
python scripts/finetune_beto.py        # Req. #3 — fine-tuning de BETO
python scripts/comparar_cnns.py        # Req. #1 — comparación de 3 CNNs
```
Los reportes se escriben en `docs/METRICAS_*.md`.
