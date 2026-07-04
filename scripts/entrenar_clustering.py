"""
entrenar_clustering.py — Entrena el modelo NO supervisado (K-Means) del plan.

Genera el dataset de AgendaFeatureVector (sintético por ahora), entrena K-Means,
lo evalúa (método del codo, silhouette, Davies-Bouldin), nombra los clusters y
guarda el artefacto. Es la evidencia de entrenamiento del módulo no supervisado.

USO:
    set PYTHONUTF8=1
    python scripts/entrenar_clustering.py

Salida:
    modelos/clustering_kmeans.pkl   → modelo entrenado + mapa id->nombre
    docs/METRICAS_clustering.md     → reporte de métricas
"""

import sys
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modulos.clustering import (
    generar_dataset, normalizar, FEATURES, ZONAS, _RUTA_MODELO,
)

_DIR = Path(__file__).resolve().parent.parent
_RUTA_REPORTE = _DIR / "docs" / "METRICAS_clustering.md"
_K = 6          # clusters fitosanitarios del plan
_SEMILLA = 42


def main():
    print("== Generando dataset de feature vectors ==")
    datos = generar_dataset(n=600, semilla=_SEMILLA)
    X = np.vstack([normalizar(d) for d in datos])
    arquetipos = [d["arquetipo_real"] for d in datos]
    print(f"  {len(datos)} diagnósticos, {X.shape[1]} features")

    # --- Método del codo + silhouette por k ---
    print("\n== Método del codo y silhouette (k=2..10) ==")
    codo = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=_SEMILLA, n_init=10).fit(X)
        sil = silhouette_score(X, km.labels_)
        codo.append((k, km.inertia_, sil))
        print(f"  k={k}: inercia={km.inertia_:7.2f}  silhouette={sil:.3f}")

    # --- Modelo final k=6 ---
    print(f"\n== Entrenando K-Means final (k={_K}) ==")
    kmeans = KMeans(n_clusters=_K, random_state=_SEMILLA, n_init=10).fit(X)
    sil = silhouette_score(X, kmeans.labels_)
    db = davies_bouldin_score(X, kmeans.labels_)
    print(f"  silhouette={sil:.3f}  davies_bouldin={db:.3f}  inercia={kmeans.inertia_:.2f}")

    # --- Nombrar cada cluster por el arquetipo mayoritario de sus miembros ---
    id2label = {}
    pureza = {}
    for cid in range(_K):
        miembros = [arquetipos[i] for i in range(len(datos)) if kmeans.labels_[i] == cid]
        if miembros:
            comun, cuenta = Counter(miembros).most_common(1)[0]
            id2label[cid] = comun
            pureza[cid] = cuenta / len(miembros)
        else:
            id2label[cid] = f"cluster_{cid}"
            pureza[cid] = 0.0
    print("\n== Mapeo cluster -> nombre (pureza) ==")
    for cid in range(_K):
        n = int((kmeans.labels_ == cid).sum())
        print(f"  cluster {cid}: {id2label[cid]:24s} n={n:3d}  pureza={pureza[cid]:.2f}")

    # --- Guardar artefacto ---
    _RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
    with open(_RUTA_MODELO, "wb") as fh:
        pickle.dump({"kmeans": kmeans, "id2label": id2label, "features": FEATURES}, fh)
    print(f"\nModelo guardado en {_RUTA_MODELO.relative_to(_DIR)}")

    # --- Demo de mapa epidemiológico: clusters por zona ---
    print("\n== Mapa epidemiológico (clusters por zona) ==")
    por_zona = {z: Counter() for z in ZONAS}
    for i, d in enumerate(datos):
        por_zona[d["zona"]][id2label[kmeans.labels_[i]]] += 1
    for z in ZONAS:
        dominante = por_zona[z].most_common(1)[0] if por_zona[z] else ("-", 0)
        print(f"  {z:10s}: dominante={dominante[0]} ({dominante[1]})")

    _escribir_reporte(codo, sil, db, kmeans, id2label, pureza, por_zona)
    print(f"Reporte en {_RUTA_REPORTE.relative_to(_DIR)}")


def _escribir_reporte(codo, sil, db, kmeans, id2label, pureza, por_zona):
    with open(_RUTA_REPORTE, "w", encoding="utf-8") as f:
        f.write("# Clustering fitosanitario — modelo NO supervisado\n\n")
        f.write("**Algoritmo:** K-Means (scikit-learn). **Tarea:** agrupar diagnósticos "
                "en clusters fitosanitarios (base del mapa epidemiológico).\n\n")
        f.write("**Datos:** 600 feature vectors (AgendaFeatureVector del plan), "
                f"{len(FEATURES)} features normalizadas [0,1]. Sintéticos por ahora "
                "(aún no hay diagnósticos reales acumulados).\n\n")
        f.write(f"**Hiperparámetros:** k={_K}, n_init=10, random_state={_SEMILLA}.\n\n")
        f.write("## Selección de k (método del codo + silhouette)\n\n")
        f.write("| k | Inercia | Silhouette |\n|---|---|---|\n")
        for k, ine, s in codo:
            f.write(f"| {k} | {ine:.2f} | {s:.3f} |\n")
        f.write(f"\n## Métricas del modelo final (k={_K})\n\n")
        f.write(f"- **Silhouette:** {sil:.3f} (clusters bien separados si → 1)\n")
        f.write(f"- **Davies-Bouldin:** {db:.3f} (mejor si → 0)\n")
        f.write(f"- **Inercia:** {kmeans.inertia_:.2f}\n\n")
        f.write("## Clusters encontrados\n\n")
        f.write("| Cluster | Nombre fitosanitario | Tamaño | Pureza |\n|---|---|---|---|\n")
        for cid in range(_K):
            n = int((kmeans.labels_ == cid).sum())
            f.write(f"| {cid} | {id2label[cid]} | {n} | {pureza[cid]:.2f} |\n")
        f.write("\n## Mapa epidemiológico (cluster dominante por zona)\n\n")
        f.write("| Zona | Cluster dominante | Casos |\n|---|---|---|\n")
        for z in ZONAS:
            dom = por_zona[z].most_common(1)[0] if por_zona[z] else ("-", 0)
            f.write(f"| {z} | {dom[0]} | {dom[1]} |\n")
        f.write("\n## Reproducir\n\n```powershell\nset PYTHONUTF8=1\n"
                "python scripts/entrenar_clustering.py\n```\n")


if __name__ == "__main__":
    main()
