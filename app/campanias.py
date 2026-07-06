"""
app/campanias.py — Mapa epidemiológico REAL desde las campañas fitosanitarias.

Lee datos/campanias/*.csv (datos reales de SENASICA) y agrega por entidad
federativa (estado) para el mapa epidemiológico que consume la app.
"""

import csv
import glob
from pathlib import Path
from collections import Counter

_DIR_BASE = Path(__file__).resolve().parent.parent
_DIR_CSV = _DIR_BASE / "datos" / "campanias"


def _num(x: str) -> float:
    try:
        return float(str(x).replace(",", "").strip())
    except (ValueError, AttributeError):
        return 0.0


def _cargar() -> list[dict]:
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
                    "superficie": _num(rr.get("sa", 0)),
                    "productores": _num(rr.get("productores_atendidos", 0)),
                })
    return filas


def mapa() -> dict:
    """
    Mapa epidemiológico real: por estado, nº de campañas, superficie total,
    campaña y cultivo dominantes. Ordenado por superficie descendente.
    """
    filas = _cargar()
    por_estado: dict = {}
    for f in filas:
        e = f["estado"] or "(sin estado)"
        d = por_estado.setdefault(e, {"campania": Counter(), "cultivo": Counter(),
                                      "superficie": 0.0, "productores": 0.0, "n": 0})
        d["campania"][f["campania"]] += 1
        if f["cultivo"]:
            d["cultivo"][f["cultivo"]] += 1
        d["superficie"] += f["superficie"]
        d["productores"] += f["productores"]
        d["n"] += 1

    estados = []
    for e, d in por_estado.items():
        camp = d["campania"].most_common(1)[0][0] if d["campania"] else ""
        cul = d["cultivo"].most_common(1)[0][0] if d["cultivo"] else ""
        estados.append({
            "estado": e,
            "campanias": d["n"],
            "superficie_ha": round(d["superficie"], 1),
            "productores": int(d["productores"]),
            "campania_dominante": camp,
            "cultivo_dominante": cul,
        })
    estados.sort(key=lambda x: x["superficie_ha"], reverse=True)
    return {"total_campanias": len(filas), "estados": estados}
