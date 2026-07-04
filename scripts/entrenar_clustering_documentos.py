"""
entrenar_clustering_documentos.py — Clustering del corpus (no supervisado).

Segundo clustering del proyecto: agrupa los fragmentos del corpus por TEMA usando
sus embeddings BERT (384-d) ya calculados. Sirve para organizar la base de
conocimiento y encontrar documentos relacionados.

A diferencia del clustering de diagnósticos (datos sintéticos), este usa DATOS
REALES: los embeddings de datos/embeddings_bert.pkl.

USO:
    set PYTHONUTF8=1
    python scripts/entrenar_clustering_documentos.py

Salida:
    modelos/clustering_documentos.pkl   → modelo K-Means
    docs/METRICAS_clustering_documentos.md
"""

import sys
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modulos.almacen_documentos import _conectar, _RUTA_BD
from modulos.busqueda_semantica import _RUTA_EMBEDDINGS

_DIR = Path(__file__).resolve().parent.parent
_RUTA_MODELO = _DIR / "modelos" / "clustering_documentos.pkl"
_RUTA_REPORTE = _DIR / "docs" / "METRICAS_clustering_documentos.md"
_SEMILLA = 42


def cargar_embeddings_y_meta():
    """Carga los embeddings y los metadatos (cultivo/enfermedad/fuente) por id."""
    if not _RUTA_EMBEDDINGS.exists():
        raise SystemExit("No hay embeddings. Ejecuta scripts/construir_corpus.py primero.")
    with open(_RUTA_EMBEDDINGS, "rb") as fh:
        datos = pickle.load(fh)
    ids = datos["ids"]
    X = np.asarray(datos["embeddings"], dtype=np.float32)

    con = _conectar(_RUTA_BD)
    meta = {}
    for doc_id in ids:
        fila = con.execute(
            "SELECT cultivo, enfermedad, fuente FROM documentos WHERE id = ?",
            (doc_id,)).fetchone()
        meta[doc_id] = dict(fila) if fila else {"cultivo": "?", "enfermedad": "?", "fuente": "?"}
    con.close()
    return ids, X, meta


def main():
    print("== Cargando embeddings del corpus ==")
    ids, X, meta = cargar_embeddings_y_meta()
    # Normalización L2: K-Means (Euclidiana) sobre vectores unitarios ≈ coseno,
    # que es la métrica natural de los embeddings BERT (spherical k-means).
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    print(f"  {X.shape[0]} documentos, {X.shape[1]} dimensiones (BERT, L2-normalizados)")

    # --- Selección de k (codo + silhouette) ---
    print("\n== Método del codo y silhouette ==")
    codo = []
    for k in range(4, 16):
        km = KMeans(n_clusters=k, random_state=_SEMILLA, n_init=10).fit(X)
        sil = silhouette_score(X, km.labels_)
        codo.append((k, km.inertia_, sil))
        print(f"  k={k:2d}: inercia={km.inertia_:8.1f}  silhouette={sil:.3f}")

    mejor_k = max(codo, key=lambda t: t[2])[0]
    print(f"\n== Mejor k por silhouette: {mejor_k} ==")

    kmeans = KMeans(n_clusters=mejor_k, random_state=_SEMILLA, n_init=10).fit(X)
    sil = silhouette_score(X, kmeans.labels_)
    db = davies_bouldin_score(X, kmeans.labels_)
    print(f"  silhouette={sil:.3f}  davies_bouldin={db:.3f}")

    # --- Describir cada cluster por su cultivo/enfermedad dominante ---
    print("\n== Temas por cluster ==")
    resumen = []
    for cid in range(mejor_k):
        idx = [i for i in range(len(ids)) if kmeans.labels_[i] == cid]
        cultivos = Counter(meta[ids[i]]["cultivo"] for i in idx)
        enfermedades = Counter(meta[ids[i]]["enfermedad"] for i in idx)
        cul = cultivos.most_common(1)[0] if cultivos else ("?", 0)
        enf = enfermedades.most_common(1)[0] if enfermedades else ("?", 0)
        etiqueta = f"{cul[0]} / {enf[0]}"
        resumen.append((cid, len(idx), etiqueta, cul, enf))
        print(f"  cluster {cid:2d}: n={len(idx):4d}  tema='{etiqueta}'  "
              f"(cultivo {cul[1]}/{len(idx)})")

    _RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
    with open(_RUTA_MODELO, "wb") as fh:
        pickle.dump({"kmeans": kmeans, "ids": ids,
                     "temas": {cid: r[2] for cid, *r in [(x[0], *x[1:]) for x in resumen]}}, fh)
    print(f"\nModelo guardado en {_RUTA_MODELO.relative_to(_DIR)}")

    _escribir_reporte(X, codo, mejor_k, sil, db, resumen)
    print(f"Reporte en {_RUTA_REPORTE.relative_to(_DIR)}")


def _escribir_reporte(X, codo, mejor_k, sil, db, resumen):
    with open(_RUTA_REPORTE, "w", encoding="utf-8") as f:
        f.write("# Clustering de documentos — modelo NO supervisado (datos reales)\n\n")
        f.write("**Algoritmo:** K-Means (scikit-learn). **Tarea:** agrupar los fragmentos "
                "del corpus por tema usando sus embeddings BERT.\n\n")
        f.write(f"**Datos:** {X.shape[0]} fragmentos reales del corpus, "
                f"embeddings de {X.shape[1]} dimensiones (paraphrase-multilingual-MiniLM-L12-v2). "
                "A diferencia del clustering de diagnósticos, aquí los datos son reales.\n\n")
        f.write("## Selección de k (codo + silhouette)\n\n")
        f.write("| k | Inercia | Silhouette |\n|---|---|---|\n")
        for k, ine, s in codo:
            marca = " ⭐" if k == mejor_k else ""
            f.write(f"| {k}{marca} | {ine:.1f} | {s:.3f} |\n")
        f.write(f"\n## Modelo final (k={mejor_k}, mejor silhouette)\n\n")
        f.write(f"- **Silhouette:** {sil:.3f}\n- **Davies-Bouldin:** {db:.3f}\n\n")
        f.write("### Interpretación honesta\n\n")
        f.write("El silhouette es **bajo** (~0.07), lo que indica que los documentos "
                "**no forman clusters bien separados**. Esto es un resultado real y "
                "esperable: el corpus es **topicamente homogéneo** (todos los fragmentos "
                "hablan de enfermedades y plagas de plantas, con vocabulario solapado), y "
                "el troceado produce fragmentos parecidos entre sí. Contrasta con el "
                "clustering de diagnósticos (silhouette ~0.49), donde los datos sí tienen "
                "estructura separable. Los grupos aquí capturan tendencias suaves (p. ej. "
                "maíz vs. calabaza vs. general), no temas nítidos. Alternativas a explorar: "
                "HDBSCAN (densidad + ruido) o reducción con UMAP antes de agrupar.\n\n")
        f.write("## Temas encontrados (cultivo/enfermedad dominante por cluster)\n\n")
        f.write("| Cluster | Tamaño | Tema dominante |\n|---|---|---|\n")
        for cid, n, etiqueta, cul, enf in resumen:
            f.write(f"| {cid} | {n} | {etiqueta} |\n")
        f.write("\n## Reproducir\n\n```powershell\nset PYTHONUTF8=1\n"
                "python scripts/entrenar_clustering_documentos.py\n```\n")


if __name__ == "__main__":
    main()
