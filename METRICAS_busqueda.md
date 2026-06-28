# Métricas del Motor de Búsqueda (Requisito #2)

> Evaluación cuantitativa y comparativa de los tres métodos de recuperación del
> sistema. Reproducible con `python evaluar_busqueda.py`.

## Metodología

- **Corpus:** 1829 fragmentos (corpus combinado: documentos del equipo + 12 PDFs).
- **Conjunto de evaluación:** 15 consultas en lenguaje natural (español), redactadas
  como las escribiría un usuario, cada una etiquetada con su **cultivo relevante**.
- **Relevancia:** un documento recuperado es relevante si su `cultivo` coincide con el
  cultivo esperado de la consulta (etiqueta automática desde los metadatos del corpus).
- **Métodos comparados:**
  1. **TF-IDF** (léxico) — `almacen_documentos.buscar`
  2. **BERT** (semántico, Sentence-BERT) — `busqueda_semantica.buscar_semantico`
  3. **Híbrido** (0.4·TF-IDF + 0.6·BERT) — `busqueda_semantica.buscar_hibrido`
- **Métricas:** P@k (precisión en top-k), Éxito@k (hit rate), MRR (rango recíproco medio).

## Resultados (top_k = 10, 15 consultas)

| Método  | P@1 | P@3 | P@5 | Éxito@3 | Éxito@5 | **MRR** |
|---------|-----|-----|-----|---------|---------|---------|
| **TF-IDF**  | **0.667** | **0.489** | **0.533** | **0.800** | **0.867** | **0.747** |
| BERT    | 0.400 | 0.444 | 0.400 | 0.733 | 0.800 | 0.536 |
| Híbrido | 0.467 | 0.444 | 0.440 | 0.733 | 0.800 | 0.601 |

**Mejor método por MRR: TF-IDF.**

## Interpretación (honesta)

- **TF-IDF gana en este test** porque las consultas suelen **nombrar el cultivo**
  ("maíz", "tomate", "calabaza"…) y TF-IDF empareja esa palabra de forma exacta con los
  fragmentos de ese cultivo. La relevancia se mide a nivel **cultivo**, lo que favorece
  al método léxico.
- **BERT no luce aquí** por la misma razón: su fortaleza es encontrar el documento
  correcto **aunque la consulta NO use las palabras clave** (p. ej. "polvo blanco" →
  "oídio"). Esa virtud no se refleja en una métrica de relevancia a nivel cultivo.
- **El híbrido queda en medio**, arrastrado por el peso 0.6 de BERT.

## Conclusión y recomendación

- El motor **funciona**: en el 87% de las consultas hay al menos un documento del cultivo
  correcto en el top-5 (Éxito@5 de TF-IDF).
- **Oportunidad de mejora:** los pesos del híbrido (0.4/0.6) no son óptimos para
  precisión a nivel cultivo. Subir el peso de TF-IDF (p. ej. 0.6/0.4) probablemente
  mejore el híbrido. Pendiente de afinar.
- **Matiz importante:** esta evaluación mide relevancia por **cultivo**. Una evaluación
  complementaria a nivel **enfermedad** (relevante solo si acierta cultivo + enfermedad)
  mostraría mejor el valor semántico de BERT. Queda como trabajo futuro.

## Cómo reproducir

```powershell
set PYTHONUTF8=1
python construir_corpus.py     # si el almacén no está construido
python evaluar_busqueda.py
```
