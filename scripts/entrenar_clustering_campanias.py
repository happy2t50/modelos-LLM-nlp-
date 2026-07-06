"""
entrenar_clustering_campanias.py — Clustering epidemiológico con DATOS REALES.

Usa datos reales de campañas fitosanitarias de México (SENASICA) en
datos/campanias/*.csv, en lugar de datos sintéticos. Agrupa las campañas por
perfil (cultivo + escala de atención) y arma el mapa epidemiológico REAL por
entidad federativa (estado).

Columnas de los CSV:
  cf = campaña fitosanitaria | pa = plaga | ef = entidad federativa (estado)
  cha = cultivo | temporalidad = trimestre | ace = control
  sa = superficie atendida | productores_atendidos

USO:
    set PYTHONUTF8=1
    python scripts/entrenar_clustering_campanias.py

Salida:
    modelos/clustering_campanias.pkl
    docs/METRICAS_clustering_campanias.md
"""

import sys
import csv
import glob
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_DIR = Path(__file__).resolve().parent.parent
_DIR_CSV = _DIR / "datos" / "campanias"
_RUTA_MODELO = _DIR / "modelos" / "clustering_campanias.pkl"
_RUTA_REPORTE = _DIR / "docs" / "METRICAS_clustering_campanias.md"
_SEMILLA = 42


def _num(x: str) -> float:
    try:
        return float(str(x).replace(",", "").strip())
    except (ValueError, AttributeError):
        return 0.0


def cargar_csvs() -> list[dict]:
    """Lee y unifica los CSV de campañas (maneja BOM y espacios en cabeceras)."""
    filas = []
    for ruta in sorted(glob.glob(str(_DIR_CSV / "*.csv"))):
        with open(ruta, encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                rr = {(k or "").strip().lower(): (v or "").strip() for k, v in r.items()}
                filas.append({
                    "campania": rr.get("cf", ""),
                    "plaga": rr.get("pa", ""),
                    "estado": rr.get("ef", ""),
                    "cultivo": rr.get("cha", "").strip().lower(),
                    "temporalidad": rr.get("temporalidad", ""),
                    "superficie": _num(rr.get("sa", 0)),
                    "productores": _num(rr.get("productores_atendidos", 0)),
                })
    return filas


def construir_features(filas):
    """Matriz de features: log(superficie), log(productores) + one-hot de cultivo."""
    cultivos = [f["cultivo"] for f in filas]
    top = [c for c, _ in Counter(cultivos).most_common(10)]
    X = []
    for f in filas:
        num = [np.log1p(f["superficie"]), np.log1p(f["productores"])]
        onehot = [1.0 if f["cultivo"] == c else 0.0 for c in top]
        X.append(num + onehot)
    X = np.array(X, dtype=float)
    X[:, :2] = StandardScaler().fit_transform(X[:, :2])  # estandarizar solo lo numérico
    return X, top


def main():
    print("== Cargando campañas fitosanitarias reales ==")
    filas = cargar_csvs()
    print(f"  {len(filas)} registros reales | estados: {len(set(f['estado'] for f in filas))}")
    X, top_cultivos = construir_features(filas)

    print("\n== Selección de k (silhouette) ==")
    codo = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=_SEMILLA, n_init=10).fit(X)
        s = silhouette_score(X, km.labels_)
        codo.append((k, km.inertia_, s))
        print(f"  k={k}: inercia={km.inertia_:7.1f}  silhouette={s:.3f}")
    mejor_k = max(codo, key=lambda t: t[2])[0]

    print(f"\n== Modelo final (k={mejor_k}) ==")
    km = KMeans(n_clusters=mejor_k, random_state=_SEMILLA, n_init=10).fit(X)
    sil = silhouette_score(X, km.labels_)
    db = davies_bouldin_score(X, km.labels_)
    print(f"  silhouette={sil:.3f}  davies_bouldin={db:.3f}")

    # Nombrar cada cluster por su campaña dominante
    id2label = {}
    for cid in range(mejor_k):
        camp = Counter(filas[i]["campania"] for i in range(len(filas)) if km.labels_[i] == cid)
        id2label[cid] = camp.most_common(1)[0][0] if camp else f"cluster_{cid}"
        n = int((km.labels_ == cid).sum())
        sup = sum(filas[i]["superficie"] for i in range(len(filas)) if km.labels_[i] == cid)
        print(f"  cluster {cid}: n={n:3d}  campaña='{id2label[cid][:35]}'  sup={sup:,.0f} ha")

    _RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
    with open(_RUTA_MODELO, "wb") as fh:
        pickle.dump({"kmeans": km, "id2label": id2label, "top_cultivos": top_cultivos}, fh)

    # Mapa epidemiológico REAL: por estado, campaña dominante + superficie total
    print("\n== Mapa epidemiológico REAL (por estado) ==")
    por_estado = {}
    for i, f in enumerate(filas):
        e = f["estado"] or "(sin estado)"
        d = por_estado.setdefault(e, {"camp": Counter(), "sup": 0.0, "n": 0})
        d["camp"][f["campania"]] += 1
        d["sup"] += f["superficie"]
        d["n"] += 1
    orden = sorted(por_estado.items(), key=lambda kv: kv[1]["sup"], reverse=True)
    for e, d in orden[:12]:
        dom = d["camp"].most_common(1)[0][0]
        print(f"  {e:22s} {d['n']:3d} campañas  {d['sup']:>12,.0f} ha  dom='{dom[:30]}'")

    _reporte(filas, codo, mejor_k, sil, db, km, id2label, por_estado)
    print(f"\nReporte en {_RUTA_REPORTE.relative_to(_DIR)}")


def _reporte(filas, codo, mejor_k, sil, db, km, id2label, por_estado):
    with open(_RUTA_REPORTE, "w", encoding="utf-8") as f:
        f.write("# Clustering epidemiológico — DATOS REALES (campañas fitosanitarias)\n\n")
        f.write("**Algoritmo:** K-Means (scikit-learn). **Datos:** REALES — campañas "
                "fitosanitarias de México (SENASICA), `datos/campanias/*.csv`.\n\n")
        f.write(f"**Registros:** {len(filas)} campañas en "
                f"{len(set(x['estado'] for x in filas))} estados. "
                "Features: log(superficie atendida), log(productores) + one-hot de cultivo.\n\n")
        f.write("> A diferencia del clustering de diagnósticos (sintético), este usa "
                "**datos reales** con entidad federativa (estado), lo que permite un "
                "**mapa epidemiológico real**.\n\n")
        f.write("## Selección de k\n\n| k | Inercia | Silhouette |\n|---|---|---|\n")
        for k, ine, s in codo:
            marca = " ⭐" if k == mejor_k else ""
            f.write(f"| {k}{marca} | {ine:.1f} | {s:.3f} |\n")
        f.write(f"\n## Modelo final (k={mejor_k})\n\n- **Silhouette:** {sil:.3f}\n"
                f"- **Davies-Bouldin:** {db:.3f}\n\n")
        f.write("## Clusters (campaña dominante)\n\n| Cluster | Campaña dominante | Tamaño | Superficie (ha) |\n|---|---|---|---|\n")
        for cid in range(mejor_k):
            n = int((km.labels_ == cid).sum())
            sup = sum(filas[i]["superficie"] for i in range(len(filas)) if km.labels_[i] == cid)
            f.write(f"| {cid} | {id2label[cid]} | {n} | {sup:,.0f} |\n")
        f.write("\n## Mapa epidemiológico real (por estado)\n\n")
        f.write("| Estado | Campañas | Superficie (ha) | Campaña dominante |\n|---|---|---|---|\n")
        orden = sorted(por_estado.items(), key=lambda kv: kv[1]["sup"], reverse=True)
        for e, d in orden:
            f.write(f"| {e} | {d['n']} | {d['sup']:,.0f} | {d['camp'].most_common(1)[0][0]} |\n")
        f.write("\n## Reproducir\n\n```powershell\nset PYTHONUTF8=1\n"
                "python scripts/entrenar_clustering_campanias.py\n```\n")


if __name__ == "__main__":
    main()
