# Clustering epidemiológico — DATOS REALES (campañas fitosanitarias)

**Algoritmo:** K-Means (scikit-learn). **Datos:** REALES — campañas fitosanitarias de México (SENASICA), `datos/campanias/*.csv`.

**Registros:** 584 campañas en 39 estados. Features: log(superficie atendida), log(productores) + one-hot de cultivo.

> A diferencia del clustering de diagnósticos (sintético), este usa **datos reales** con entidad federativa (estado), lo que permite un **mapa epidemiológico real**.

## Selección de k

| k | Inercia | Silhouette |
|---|---|---|
| 2 ⭐ | 860.5 | 0.359 |
| 3 | 669.1 | 0.272 |
| 4 | 602.8 | 0.199 |
| 5 | 555.5 | 0.207 |
| 6 | 513.8 | 0.230 |
| 7 | 484.6 | 0.238 |
| 8 | 445.3 | 0.256 |
| 9 | 422.8 | 0.253 |
| 10 | 396.1 | 0.283 |

## Modelo final (k=2)

- **Silhouette:** 0.359
- **Davies-Bouldin:** 1.046

## Clusters (campaña dominante)

| Cluster | Campaña dominante | Tamaño | Superficie (ha) |
|---|---|---|---|
| 0 | Plagas de los Cítricos | 304 | 14,639 |
| 1 | Manejo Fitosanitario del Maíz | 280 | 567,248 |

## Mapa epidemiológico real (por estado)

| Estado | Campañas | Superficie (ha) | Campaña dominante |
|---|---|---|---|
| Chihuahua | 7 | 170,195 | Plagas Reglamentadas del Algodonero |
| Sinaloa | 18 | 79,152 | Manejo Fitosanitario del Maíz |
| Sonora | 30 | 43,504 | Plagas de los Cítricos |
| Michoacán | 23 | 33,176 | Plagas de los Cítricos |
| Tabasco | 26 | 32,664 | Plagas de los Cítricos |
| Chiapas | 48 | 29,423 | Plagas de los Cítricos |
| Baja California | 12 | 19,936 | Plagas de los Cítricos |
| Veracruz | 41 | 19,306 | Campaña contra la Langosta centroamericana |
| Puebla | 38 | 18,822 | Plagas de los Cítricos |
| Oaxaca | 39 | 13,831 | Campaña contra la Langosta centroamericana |
| Tamaulipas | 25 | 12,072 | Campaña contra la Langosta centroamericana |
| Zacatecas | 6 | 11,016 | Manejo Fitosanitario del Frijol |
| Nayarit | 23 | 9,966 | Plagas de los Cítricos |
| Guerrero | 16 | 9,848 | Manejo Fitosanitario del Maíz |
| Yucatán | 19 | 9,718 | Plagas de los Cítricos |
| Nuevo León | 15 | 8,856 | Plagas de los Cítricos |
| San Luis Potosí | 22 | 7,752 | Campaña contra la Langosta centroamericana |
| Morelos | 19 | 5,619 | Plagas de los Cítricos |
| Baja California Sur | 16 | 5,506 | Plagas de los Cítricos |
| Coahuila | 7 | 5,088 | Plagas Reglamentadas del Algodonero |
| Querétaro | 3 | 5,038 | Manejo Fitosanitario del Maíz |
| Quintana Roo | 18 | 4,856 | Plagas de los Cítricos |
| Hidalgo | 10 | 4,489 | Manejo Fitosanitario del Maíz |
| Jalisco | 16 | 4,114 | Plagas de los Cítricos |
| Campeche | 29 | 3,898 | Campaña contra la Langosta centroamericana |
| Campache | 9 | 3,430 | Plagas de los Cítricos |
| Durango | 5 | 2,007 | Manejo Fitosanitario del Maíz |
| México | 3 | 1,885 | Manejo Fitosanitario del Maíz |
| Guanajuato | 3 | 1,379 | Manejo Fitosanitario del Maíz |
| San Luis Potosi | 10 | 1,215 | Campaña contra la Langosta centroamericana |
| Ciudad de México | 3 | 1,060 | Manejo Fitosanitario del Maíz |
| Tlaxcala | 2 | 927 | Manejo Fitosanitario del Maíz |
| Colima | 7 | 652 | Plagas Reglamentadas del Aguacatero |
| Aguascalientes | 6 | 651 | Manejo Fitosanitario del Maíz |
| Estado de México | 5 | 525 | Plagas Reglamentadas del Aguacatero |
| Queretaro | 2 | 233 | Manejo Fitosanitario del Maíz |
| Yucatan | 1 | 29 | Campaña contra la Langosta centroamericana |
| Michoacan | 1 | 28 | Campaña contra la Langosta centroamericana |
| 0 | 1 | 22 | Manejo Fitosanitario del Maíz |

## Reproducir

```powershell
set PYTHONUTF8=1
python scripts/entrenar_clustering_campanias.py
```
