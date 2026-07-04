# Clustering fitosanitario — modelo NO supervisado

**Algoritmo:** K-Means (scikit-learn). **Tarea:** agrupar diagnósticos en clusters fitosanitarios (base del mapa epidemiológico).

**Datos:** 600 feature vectors (AgendaFeatureVector del plan), 13 features normalizadas [0,1]. Sintéticos por ahora (aún no hay diagnósticos reales acumulados).

**Hiperparámetros:** k=6, n_init=10, random_state=42.

## Selección de k (método del codo + silhouette)

| k | Inercia | Silhouette |
|---|---|---|
| 2 | 403.53 | 0.424 |
| 3 | 275.99 | 0.436 |
| 4 | 176.31 | 0.500 |
| 5 | 132.94 | 0.475 |
| 6 | 105.33 | 0.491 |
| 7 | 97.39 | 0.444 |
| 8 | 89.42 | 0.424 |
| 9 | 81.59 | 0.384 |
| 10 | 74.14 | 0.349 |

## Métricas del modelo final (k=6)

- **Silhouette:** 0.491 (clusters bien separados si → 1)
- **Davies-Bouldin:** 0.801 (mejor si → 0)
- **Inercia:** 105.33

## Clusters encontrados

| Cluster | Nombre fitosanitario | Tamaño | Pureza |
|---|---|---|---|
| 0 | monitoreo_sin_datos | 100 | 1.00 |
| 1 | alto_riesgo_estacional | 122 | 0.82 |
| 2 | seguimiento_preventivo | 100 | 1.00 |
| 3 | reincidencia_cronica | 100 | 1.00 |
| 4 | tratamiento_prolongado | 100 | 1.00 |
| 5 | critico_activo | 78 | 1.00 |

## Mapa epidemiológico (cluster dominante por zona)

| Zona | Cluster dominante | Casos |
|---|---|---|
| Centro | monitoreo_sin_datos | 24 |
| Altos | reincidencia_cronica | 19 |
| Frailesca | alto_riesgo_estacional | 27 |
| Soconusco | tratamiento_prolongado | 24 |
| Norte | reincidencia_cronica | 21 |
| Selva | alto_riesgo_estacional | 24 |

## Reproducir

```powershell
set PYTHONUTF8=1
python scripts/entrenar_clustering.py
```
