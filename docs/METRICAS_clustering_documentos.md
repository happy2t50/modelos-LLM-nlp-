# Clustering de documentos — modelo NO supervisado (datos reales)

**Algoritmo:** K-Means (scikit-learn). **Tarea:** agrupar los fragmentos del corpus por tema usando sus embeddings BERT.

**Datos:** 1832 fragmentos reales del corpus, embeddings de 384 dimensiones (paraphrase-multilingual-MiniLM-L12-v2). A diferencia del clustering de diagnósticos, aquí los datos son reales.

## Selección de k (codo + silhouette)

| k | Inercia | Silhouette |
|---|---|---|
| 4 ⭐ | 902.5 | 0.075 |
| 5 | 874.7 | 0.053 |
| 6 | 853.0 | 0.057 |
| 7 | 833.2 | 0.059 |
| 8 | 816.8 | 0.057 |
| 9 | 805.4 | 0.051 |
| 10 | 792.0 | 0.062 |
| 11 | 781.3 | 0.049 |
| 12 | 769.8 | 0.046 |
| 13 | 759.6 | 0.047 |
| 14 | 750.5 | 0.054 |
| 15 | 743.0 | 0.053 |

## Modelo final (k=4, mejor silhouette)

- **Silhouette:** 0.075
- **Davies-Bouldin:** 2.913

### Interpretación honesta

El silhouette es **bajo** (~0.07), lo que indica que los documentos **no forman clusters bien separados**. Esto es un resultado real y esperable: el corpus es **topicamente homogéneo** (todos los fragmentos hablan de enfermedades y plagas de plantas, con vocabulario solapado), y el troceado produce fragmentos parecidos entre sí. Contrasta con el clustering de diagnósticos (silhouette ~0.49), donde los datos sí tienen estructura separable. Los grupos aquí capturan tendencias suaves (p. ej. maíz vs. calabaza vs. general), no temas nítidos. Alternativas a explorar: HDBSCAN (densidad + ruido) o reducción con UMAP antes de agrupar.

## Temas encontrados (cultivo/enfermedad dominante por cluster)

| Cluster | Tamaño | Tema dominante |
|---|---|---|
| 0 | 575 | maíz / plagas y enfermedades |
| 1 | 632 | general / enfermedades y plagas |
| 2 | 402 | calabaza / enfermedades y plagas |
| 3 | 223 | general / producción y enfermedades |

## Reproducir

```powershell
set PYTHONUTF8=1
python scripts/entrenar_clustering_documentos.py
```
