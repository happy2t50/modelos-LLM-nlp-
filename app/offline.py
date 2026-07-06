"""
app/offline.py — Datos para el modo offline de la app (puntos 🔌 1 y 2 del plan).

Expone el corpus (datos/almacen.db) y sus embeddings (datos/embeddings_bert.pkl)
para que el teléfono los descargue y haga RAG on-device sin internet.

Un "documento offline" = un grupo de fragmentos que comparten (cultivo, enfermedad,
fuente). Sus "chunks" son esos fragmentos, cada uno con su embedding (384-d).

Ref.: offline_architecture.md §6 (Punto 1: catálogo; Punto 2: descarga con chunks).
"""

import pickle
import hashlib
from typing import Optional
from collections import Counter

import numpy as np

from modulos.almacen_documentos import _conectar, _RUTA_BD
from modulos.busqueda_semantica import _RUTA_EMBEDDINGS


def _doc_id(cultivo: str, fuente: str) -> str:
    """Id estable y determinista para un documento (una fuente por cultivo)."""
    clave = f"{cultivo}|{fuente}"
    return "doc_" + hashlib.md5(clave.encode("utf-8")).hexdigest()[:12]


def _cargar_grupos() -> dict:
    """
    Agrupa los fragmentos por (cultivo, fuente) = un documento fuente por cultivo.
    La enfermedad del documento es la etiqueta más común entre sus fragmentos.
    """
    con = _conectar(_RUTA_BD)
    filas = con.execute(
        "SELECT id, cultivo, enfermedad, fuente, texto FROM documentos "
        "ORDER BY cultivo, fuente, fragmento"
    ).fetchall()
    con.close()

    grupos: dict = {}
    for f in filas:
        did = _doc_id(f["cultivo"], f["fuente"])
        g = grupos.setdefault(did, {
            "cultivo": f["cultivo"], "fuente": f["fuente"],
            "enf": Counter(), "frag": [],
        })
        g["enf"][f["enfermedad"] or ""] += 1
        g["frag"].append((f["id"], f["texto"] or ""))

    # Enfermedad representativa = la más frecuente del grupo
    for g in grupos.values():
        g["enfermedad"] = g["enf"].most_common(1)[0][0] if g["enf"] else ""
    return grupos


# Cache del mapa id_fragmento -> embedding
_emb_cache: Optional[dict] = None


def _emb_map() -> dict:
    global _emb_cache
    if _emb_cache is None:
        if not _RUTA_EMBEDDINGS.exists():
            _emb_cache = {}
        else:
            with open(_RUTA_EMBEDDINGS, "rb") as fh:
                d = pickle.load(fh)
            _emb_cache = {
                int(i): np.asarray(e, dtype=float)
                for i, e in zip(d["ids"], d["embeddings"])
            }
    return _emb_cache


def _tam_bytes(fragmentos) -> int:
    return sum(len(t.encode("utf-8")) for _, t in fragmentos)


def catalogo() -> dict:
    """Catálogo de documentos disponibles para descarga offline (🔌 Punto 1)."""
    grupos = _cargar_grupos()
    docs = []
    for did, g in grupos.items():
        title = f"{g['cultivo'].capitalize()} — {g['enfermedad']}"
        docs.append({
            "id": did,
            "crop_name": g["cultivo"],
            "disease_name": g["enfermedad"],
            "title": title,
            "source": g["fuente"] or "sin fuente",
            "size_bytes": _tam_bytes(g["frag"]),
            "version": "1.0",
        })
    docs.sort(key=lambda d: (d["crop_name"], d["disease_name"], d["source"]))
    return {"documents": docs}


def documento(doc_id: str) -> Optional[dict]:
    """
    Documento con su contenido, chunks y embeddings para descarga (🔌 Punto 2).
    Los embeddings son de 384 dimensiones (MiniLM-L12), no 768.
    """
    grupos = _cargar_grupos()
    g = grupos.get(doc_id)
    if g is None:
        return None

    emb = _emb_map()
    chunks = []
    vecs = []
    for idx, (row_id, texto) in enumerate(g["frag"]):
        v = emb.get(int(row_id))
        vec_list = v.tolist() if v is not None else []
        if v is not None:
            vecs.append(v)
        chunks.append({
            "id": f"{doc_id}_c{idx}",
            "index": idx,
            "text": texto,
            "embedding": vec_list,
        })

    global_emb = np.mean(vecs, axis=0).tolist() if vecs else []
    return {
        "id": doc_id,
        "content": "\n\n".join(t for _, t in g["frag"]),
        "size_bytes": _tam_bytes(g["frag"]),
        "embedding": global_emb,
        "chunks": chunks,
    }
