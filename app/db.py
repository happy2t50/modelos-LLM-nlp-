"""
app/db.py — Persistencia del historial de inferencias (SQLite).

Cumple el requisito de "almacenar las inferencias" y exponer un historial.
Se usa una BD separada de la del corpus (almacen.db) para no mezclar dominios.
"""

import json
import uuid
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.config import RUTA_DB_INFERENCIAS


def _conectar() -> sqlite3.Connection:
    RUTA_DB_INFERENCIAS.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(RUTA_DB_INFERENCIAS))
    con.row_factory = sqlite3.Row
    con.execute("""
        CREATE TABLE IF NOT EXISTS inferencias (
            id            TEXT PRIMARY KEY,
            created_at    TEXT NOT NULL,
            cultivo       TEXT,
            enfermedad    TEXT,
            confianza     REAL,
            modo          TEXT,
            latency_ms    INTEGER,
            request_json  TEXT,
            response_json TEXT
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_inf_fecha ON inferencias (created_at DESC)")
    con.execute("""
        CREATE TABLE IF NOT EXISTS clustering (
            id            TEXT PRIMARY KEY,
            created_at    TEXT NOT NULL,
            zona          TEXT,
            cluster_id    INTEGER,
            cluster_label TEXT,
            features_json TEXT
        )
    """)
    con.commit()
    return con


def guardar_clustering(zona: str, cluster_id: int, cluster_label: str,
                       features: dict) -> str:
    """Persiste una asignación de cluster (para el mapa epidemiológico)."""
    cid = "clu_" + uuid.uuid4().hex[:12]
    creado = datetime.now(timezone.utc).isoformat()
    con = _conectar()
    con.execute(
        "INSERT INTO clustering (id, created_at, zona, cluster_id, cluster_label, "
        "features_json) VALUES (?,?,?,?,?,?)",
        (cid, creado, zona, cluster_id, cluster_label,
         json.dumps(features, ensure_ascii=False)),
    )
    con.commit()
    con.close()
    return cid


def mapa_epidemiologico() -> dict:
    """Agrega los clusters por zona → mapa epidemiológico."""
    con = _conectar()
    total = con.execute("SELECT COUNT(*) FROM clustering").fetchone()[0]
    filas = con.execute(
        "SELECT zona, cluster_label, COUNT(*) AS n FROM clustering "
        "GROUP BY zona, cluster_label"
    ).fetchall()
    con.close()

    por_zona: dict[str, dict] = {}
    for f in filas:
        z = f["zona"] or "(sin zona)"
        por_zona.setdefault(z, {"total": 0, "conteos": {}})
        por_zona[z]["conteos"][f["cluster_label"]] = f["n"]
        por_zona[z]["total"] += f["n"]

    zonas = []
    for z, info in sorted(por_zona.items()):
        dom = max(info["conteos"].items(), key=lambda kv: kv[1]) if info["conteos"] else ("-", 0)
        zonas.append({"zona": z, "cluster_dominante": dom[0],
                      "casos": dom[1], "total": info["total"]})
    return {"total": total, "zonas": zonas}


def guardar_inferencia(request: dict, response: dict, modo: str,
                       latency_ms: int) -> tuple[str, str]:
    """Persiste una inferencia y devuelve (id, created_at ISO)."""
    inf_id = "inf_" + uuid.uuid4().hex[:12]
    creado = datetime.now(timezone.utc).isoformat()
    con = _conectar()
    con.execute(
        "INSERT INTO inferencias (id, created_at, cultivo, enfermedad, confianza, "
        "modo, latency_ms, request_json, response_json) VALUES (?,?,?,?,?,?,?,?,?)",
        (inf_id, creado,
         request.get("cropName", ""), request.get("diseaseName", ""),
         float(response.get("confianzaAjustada", 0.0)),
         modo, latency_ms,
         json.dumps(request, ensure_ascii=False),
         json.dumps(response, ensure_ascii=False)),
    )
    con.commit()
    con.close()
    return inf_id, creado


def listar_inferencias(limit: int = 20, offset: int = 0) -> dict:
    """Devuelve el historial paginado."""
    con = _conectar()
    total = con.execute("SELECT COUNT(*) FROM inferencias").fetchone()[0]
    filas = con.execute(
        "SELECT id, created_at, cultivo, enfermedad, confianza FROM inferencias "
        "ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset),
    ).fetchall()
    con.close()
    items = [{
        "inference_id": f["id"], "created_at": f["created_at"],
        "cultivo": f["cultivo"], "enfermedad": f["enfermedad"],
        "confianza": f["confianza"],
    } for f in filas]
    return {"total": total, "items": items}


def obtener_inferencia(inf_id: str) -> dict | None:
    """Detalle completo de una inferencia (request + response)."""
    con = _conectar()
    f = con.execute("SELECT * FROM inferencias WHERE id = ?", (inf_id,)).fetchone()
    con.close()
    if f is None:
        return None
    return {
        "inference_id": f["id"], "created_at": f["created_at"],
        "cultivo": f["cultivo"], "enfermedad": f["enfermedad"],
        "confianza": f["confianza"], "modo": f["modo"],
        "latency_ms": f["latency_ms"],
        "request": json.loads(f["request_json"]),
        "response": json.loads(f["response_json"]),
    }
