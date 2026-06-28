# Plan de trabajo v2 — Sistema de Diagnóstico Agrícola (para Claude Code)

Alineado con el documento de arquitectura del equipo (`flujo.md`) y las decisiones acordadas.

## Decisiones fijadas

- **Un solo modelo CNN** (EfficientNet-B4, 50 clases, ya entrenado al ~97%). NO se construyen 15 modelos por cultivo; el modelo único predice cultivo+enfermedad junto (p. ej. `Tomato___Late_blight`).
- **Funcionamiento online/offline con degradación** (para el campo, sin conexión fiable).
- **Búsqueda híbrida TF-IDF + BERT** (Sentence-BERT).
- Entrada **imagen + texto** (CNN + NLP fusionados).
- **"Mis cultivos"**: función donde el usuario registra los cultivos de su parcela. Filtra qué se pre-carga en caché y qué se busca.
- **Caché en SQLite por Top-K**: al consultar con internet, se guardan los K (10) mejores documentos por enfermedad. Se pre-carga ("semilla") según los cultivos registrados, para que el modo offline funcione desde el día uno.
- **3 roles**: agricultor (respuesta simple), aprendiz (respuesta técnica/educativa), gestor/admin (administración; ver Fase 10, exploratoria). El login/gestión de roles se maneja en el back/app móvil, fuera de este plan de Python.

> ⚠️ **Requisito de Minería:** se quitó el clustering. Posible reemplazo natural: el **mapa epidemiológico** del gestor (Fase 10) es minería de datos real (agrupación de reportes por zona/condiciones). Confírmenlo con el profesor.

---

## Cómo funciona el caché (la pieza clave del offline)

```
INSTALACIÓN / "Mis cultivos" (online, una vez):
  usuario registra: [maíz, calabaza, frijol]
    → descarga el TOP-K de documentos de las enfermedades de ESOS cultivos
    → los guarda en SQLite  (caché semilla, filtrado por parcela)

USO — CON internet:
  foto + texto → CNN+NLP → diagnóstico
    → IR busca fresco (filtrado a los cultivos del usuario)
    → toma TOP-K (10) → los GUARDA/actualiza en SQLite → responde

USO — SIN internet:
  foto + texto → CNN+NLP → diagnóstico
    → busca en SQLite (el TOP-K ya guardado) con TF-IDF + BERT
    → responde, indicando modo offline
```

El Top-K es el mecanismo de guardado (solo lo más relevante, compacto). "Mis cultivos" es el filtro que mantiene el caché ligero (solo los cultivos del usuario, no los 15).

---

## Componentes (módulos)

1. **CNN** (`clasificador.py`) — foto → cultivo+enfermedad. **Ya hecho** (best.pth).
2. **NLP** (`nlp_texto.py`) — interpreta el texto del usuario (síntomas).
3. **Fusión** (`fusion.py`) — combina CNN + NLP en un diagnóstico.
4. **Mis cultivos** (`mis_cultivos.py`) — registra/edita los cultivos de la parcela (tabla SQLite).
5. **Almacén** (`almacen_documentos.py`) — SQLite + TF-IDF, con caché Top-K.
6. **BERT** (`busqueda_semantica.py`) — Sentence-BERT + búsqueda híbrida.
7. **Conexión** (`conexion.py`) — detecta internet; lógica online/offline.
8. **LLM** (`generador.py`) — Qwen vía Ollama; respuesta según rol.
9. **Productos** (`productos.py`) — catálogo (web scraping). Opcional/frágil.
10. **Orquestador** (`asistente.py`) — une todo.

> **Nota offline:** sin internet el móvil no alcanza ningún servidor. Qwen y BERT en modo offline deben correr en el dispositivo o en una laptop local de la zona. Ambos son pesados para un teléfono → define esto antes del empaquetado (Fase 9).

---

## Cómo trabajar con Claude Code

- **Una tarea por vez**; revisa antes de seguir.
- **Prueba cada fase** antes de pasar a la siguiente.
- **git desde el inicio** (`git init`) y un commit al cerrar cada fase.
- **No pidas "hazme todo".** El valor está en ir por partes.
- Si algo falla, pégale el error completo y pide que lo explique antes de arreglar.

---

## FASE 0 — Preparación

> "Lee el CLAUDE.md. Inicializa git, crea la estructura (modulos/, datos/, documentos/, tests/) y un entorno virtual con requirements.txt inicial. Solo la estructura, no instales nada aún."

---

## FASE 1 — Almacén de documentos + caché Top-K (SQLite + TF-IDF) ⭐ núcleo offline

> "Crea `almacen_documentos.py` que: (1) guarde documentos (texto + metadatos: cultivo, enfermedad, fuente) en SQLite; (2) construya un índice TF-IDF con scikit-learn; (3) tenga `buscar(consulta, cultivos=None, top_k=10)` que devuelva los K más relevantes, opcionalmente filtrando por una lista de cultivos; (4) tenga `guardar_topk(enfermedad, documentos)` para cachear los mejores K de una consulta; (5) cargue documentos desde documentos/ (.txt y .pdf). Añade un script de prueba con 2-3 documentos."

**Probar:** "tizón tardío tomate" devuelve el documento correcto; el filtrado por cultivo funciona.

---

## FASE 2 — Búsqueda semántica con BERT (Sentence-BERT)

Requisito: `pip install sentence-transformers`. Modelo multilingüe (entiende español): `paraphrase-multilingual-MiniLM-L12-v2`. Descarga una vez, luego local/offline.

> "Crea `busqueda_semantica.py` con Sentence-BERT (modelo paraphrase-multilingual-MiniLM-L12-v2): calcula y guarda en disco los embeddings de los documentos del almacén; búsqueda por coseno. Crea `buscar_hibrido(consulta, cultivos=None, top_k=10)` que combine y pondere TF-IDF (fase 1) y BERT, respetando el filtro por cultivos."

**Probar:** "hongo blanco en hojas" encuentra "oídio/powdery mildew" aunque no diga esas palabras.

---

## FASE 3 — "Mis cultivos" (registro de la parcela)

> "Crea `mis_cultivos.py` que gestione, en una tabla SQLite `mis_cultivos`, la lista de cultivos de la parcela del usuario: funciones para agregar, quitar y listar cultivos. Esta lista se usará para filtrar búsquedas y para pre-cargar el caché. Sin sistema de cuentas; es configuración local de la app."

**Probar:** agregar maíz/calabaza/frijol y listarlos.

---

## FASE 4 — NLP del texto del usuario

> "Crea `nlp_texto.py` que procese el texto breve del usuario (ej. 'hojas amarillas y polvo blanco'): limpieza (minúsculas, quitar caracteres especiales), tokenización y extracción de síntomas/palabras clave. Devuelve una lista de síntomas normalizada para enriquecer la búsqueda. Usa NLP ligero en español; evita dependencias pesadas innecesarias."

**Probar:** frases de ejemplo producen los síntomas esperados.

---

## FASE 5 — Fusión CNN + NLP

> "Crea `fusion.py` con `combinar(resultado_cnn, sintomas_nlp)` que una el diagnóstico de imagen (cultivo+enfermedad+confianza) con los síntomas del texto, para formar una consulta enriquecida para el IR y ajustar la confianza si el texto refuerza o contradice la predicción visual. Lógica simple y explicada."

**Probar:** CNN "Mildiu 0.72" + texto "hojas amarillas y humedad" → diagnóstico reforzado.

---

## FASE 6 — Generador de respuesta (Qwen) + rol del usuario

> "Crea `generador.py` con `responder(diagnostico, sintomas, documentos, rol)` que arme un prompt con los documentos recuperados y lo envíe a Qwen (Ollama, API HTTP localhost:11434). Reglas: responder SOLO con base en los documentos; si no está, decir 'no tengo esa información'; nunca inventar dosis. Si rol='agricultor' → respuesta simple y directa; si rol='aprendiz' → explicación técnica y educativa. Devuelve diagnóstico, tratamiento, prevención y fuentes."

**Probar:** misma consulta como agricultor vs aprendiz da respuestas de distinto nivel.

> **RAG completo con imagen + texto + rol.** Commit.

---

## FASE 7 — Conexión + lógica online/offline + caché Top-K ⭐ diferenciador

> "Crea `conexion.py` que detecte internet (timeout corto). Crea el orquestador `asistente.py` con `consultar(imagen, texto, rol)` que: lea los cultivos de 'mis_cultivos'; si HAY internet → busca fresco (filtrado por cultivos), guarda el TOP-K en SQLite y responde; si NO hay internet → usa la búsqueda híbrida local (TF-IDF + BERT) sobre el caché. Añade `precargar_cache()` que, con internet, descargue y guarde el TOP-K de las enfermedades de los cultivos registrados (caché semilla). La salida indica el modo (online/offline)."

**Probar:** precargar con maíz/calabaza/frijol; luego desconectar y verificar que responde offline solo de esos cultivos.

---

## FASE 8 — Conectar el clasificador de imágenes (tu CNN)

> "Integra el clasificador existente (best.pth, EfficientNet-B4, 50 clases) como `clasificador.py` con `predecir(ruta_imagen)` → cultivo+enfermedad y confianza (class_mapping viene en el best.pth). Conéctalo al orquestador. Si confianza < 50%, avisa que la foto puede no ser válida. Si la enfermedad detectada es de un cultivo que NO está en 'mis cultivos', avisa al usuario."

**Probar:** foto de hoja → diagnóstico + tratamiento end-to-end, online y offline.

> **Sistema central completo.** Commit.

---

## FASE 9 — Productos (web scraping) · opcional / frágil

> "Crea `productos.py` que extraiga (scraping) un catálogo de productos fitosanitarios (nombre, marca, ingrediente activo, dosis, aplicación, para qué enfermedad) de UNA fuente indicada, guardándolo en SQLite. Respeta el sitio (pausas, robots.txt). Añade `productos_para(enfermedad)`."

**Advertencias:** el scraping se rompe si el sitio cambia; revisa términos de uso; empieza con una fuente. Alternativa robusta: tabla CSV hecha a mano.

---

## FASE 10 — Gestor / mapa epidemiológico · EXPLORATORIA (no construir aún)

Idea del equipo, todavía en definición. **No se desarrolla hasta que el núcleo (fases 1-8) funcione y la idea madure.**

Objetivo: el gestor genera un **mapa epidemiológico** (dónde y bajo qué condiciones —humedad, zona— aparecen brotes) y emite alertas a los agricultores de esa zona.

**Dependencias que el sistema actual NO tiene todavía (tenerlas en cuenta):**
- Ubicación (GPS) de los usuarios.
- Reporte de diagnósticos a un **servidor central** (hoy todo es local/offline).
- Varios usuarios reportando (sin datos no hay mapa).
- Conexión (un mapa de brotes es inherentemente online).

> Esta fase podría cubrir el **requisito de Minería** (agrupación de reportes por zona/condiciones = análisis de patrones). Confírmenlo con el profesor.

---

## Orden y dependencias

```
Fase 0 (setup)
 └ Fase 1 (SQLite + TF-IDF + Top-K)   ← núcleo offline
   └ Fase 2 (BERT)                    ← búsqueda híbrida
     └ Fase 3 (Mis cultivos)
       └ Fase 4 (NLP texto)
         └ Fase 5 (fusión CNN+NLP)
           └ Fase 6 (Qwen + rol)       ✔ RAG completo
             └ Fase 7 (online/offline + caché)  ✔ diferenciador
               └ Fase 8 (CNN imagen)   ✔ sistema central
                 ├ Fase 9 (productos)  opcional
                 └ Fase 10 (gestor/mapa) EXPLORATORIA
```

---

## Consejos finales

- **No mezcles fases.** Una a la vez, probada.
- **Los documentos son lo más importante.** El sistema solo es tan bueno como los textos que le des sobre tus 50 enfermedades.
- **El modo offline es tu diferenciador** y lo más difícil; por eso va temprano (fases 1-7).
- **Confirma el requisito de minería** (posiblemente el mapa epidemiológico).
- **Commits frecuentes**, uno por fase que funcione.
