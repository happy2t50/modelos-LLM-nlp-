"""
clustering.py — Clustering fitosanitario (aprendizaje NO supervisado).

Implementa el AgendaFeatureVector del plan (create_cloustering_agenda.md):
cada diagnóstico se convierte en un vector de características; K-Means agrupa los
diagnósticos en clusters fitosanitarios. Sirve además de base para el
**mapa epidemiológico** (agregación de clusters por zona).

Contiene:
  - Definición y normalización del feature vector.
  - Generación de datos sintéticos realistas (aún no hay diagnósticos reales).
  - Predicción de cluster para un vector (carga el modelo entrenado).
  - Agregación por zona para el mapa epidemiológico.
"""

import math
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

_DIR_BASE = Path(__file__).resolve().parent.parent
_RUTA_MODELO = _DIR_BASE / "modelos" / "clustering_kmeans.pkl"

# Orden fijo de las features numéricas que entran al modelo (mismo en train e inferencia)
FEATURES = [
    "cnnConfidence", "topKEntropy",
    "llmConfianzaAjustada", "llmEstado", "llmSinDocumentos",
    "sintomasCount", "avisosCount", "diagnosticoLength",
    "completionRate", "daysSinceCreation", "isComplete", "hasRelapse",
    "diagnosisMonth",
]

# Nombres de los 6 clusters fitosanitarios (del plan §6.4). El mapeo real
# cluster_id -> nombre se fija tras entrenar, según los centroides.
CLUSTERS_FITOSANITARIOS = [
    "critico_activo", "seguimiento_preventivo", "tratamiento_prolongado",
    "monitoreo_sin_datos", "alto_riesgo_estacional", "reincidencia_cronica",
]

# Zonas (para el mapa epidemiológico). Sintéticas por ahora (regiones de Chiapas).
ZONAS = ["Centro", "Altos", "Frailesca", "Soconusco", "Norte", "Selva"]


# ─────────────────────────────────────────────
# Normalización de features (min-max con clips, del plan §6.3)
# ─────────────────────────────────────────────

def normalizar(vec: dict) -> np.ndarray:
    """Convierte un dict de features crudas en el vector numérico normalizado [0,1]."""
    f = {}
    f["cnnConfidence"] = _clip01(vec.get("cnnConfidence", 0.0))
    f["topKEntropy"] = _clip01(vec.get("topKEntropy", 0.0) / math.log2(5))  # entropía máx top-5
    f["llmConfianzaAjustada"] = _clip01(vec.get("llmConfianzaAjustada", 0.0))
    f["llmEstado"] = _clip01(vec.get("llmEstado", 0.0))                 # 0 | 0.5 | 1
    f["llmSinDocumentos"] = _clip01(vec.get("llmSinDocumentos", 0.0))   # 0 | 1
    f["sintomasCount"] = _clip01(vec.get("sintomasCount", 0.0) / 10.0)
    f["avisosCount"] = _clip01(vec.get("avisosCount", 0.0) / 5.0)
    f["diagnosticoLength"] = _clip01(vec.get("diagnosticoLength", 0.0) / 1000.0)
    f["completionRate"] = _clip01(vec.get("completionRate", 0.0))
    f["daysSinceCreation"] = _clip01(vec.get("daysSinceCreation", 0.0) / 90.0)
    f["isComplete"] = _clip01(vec.get("isComplete", 0.0))
    f["hasRelapse"] = _clip01(vec.get("hasRelapse", 0.0))
    f["diagnosisMonth"] = _clip01((vec.get("diagnosisMonth", 1) - 1) / 11.0)
    return np.array([f[k] for k in FEATURES], dtype=np.float32)


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


# ─────────────────────────────────────────────
# Generación de datos sintéticos (arquetipos → clusters naturales)
# ─────────────────────────────────────────────

# Cada arquetipo define medias de features que producen un cluster reconocible.
_ARQUETIPOS = {
    "critico_activo":        dict(cnnConfidence=0.93, avisosCount=3.5, completionRate=0.10,
                                  llmEstado=1.0, hasRelapse=0.0, daysSinceCreation=3),
    "seguimiento_preventivo": dict(cnnConfidence=0.80, avisosCount=0.3, completionRate=1.0,
                                   llmEstado=1.0, hasRelapse=0.0, daysSinceCreation=20),
    "tratamiento_prolongado": dict(cnnConfidence=0.75, avisosCount=1.0, completionRate=0.35,
                                    llmEstado=0.5, hasRelapse=0.0, daysSinceCreation=45),
    "monitoreo_sin_datos":   dict(cnnConfidence=0.45, avisosCount=0.5, completionRate=0.20,
                                  llmEstado=0.0, llmSinDocumentos=1.0, topKEntropy=2.0,
                                  daysSinceCreation=5),
    "alto_riesgo_estacional": dict(cnnConfidence=0.85, avisosCount=2.0, completionRate=0.30,
                                    llmEstado=1.0, diagnosisMonth=12, daysSinceCreation=8),
    "reincidencia_cronica":  dict(cnnConfidence=0.82, avisosCount=1.5, completionRate=0.50,
                                  llmEstado=1.0, hasRelapse=1.0, daysSinceCreation=60),
}

_CULTIVOS = ["tomate", "calabaza", "maíz", "frijol", "papa", "chile"]
_ENFERMEDADES = ["tizón tardío", "oídio", "roya", "mancha foliar", "mosaico", "antracnosis"]


def generar_dataset(n: int = 600, semilla: int = 42) -> list[dict]:
    """
    Genera n diagnósticos sintéticos repartidos entre los 6 arquetipos, con ruido.
    Cada registro trae features crudas + metadatos (cultivo, enfermedad, zona).
    """
    rng = np.random.default_rng(semilla)
    nombres = list(_ARQUETIPOS.keys())
    registros = []
    for i in range(n):
        arq = nombres[i % len(nombres)]
        base = _ARQUETIPOS[arq]
        reg = {
            "diagnosisId": f"syn_{i:04d}",
            "cropName": _CULTIVOS[rng.integers(len(_CULTIVOS))],
            "diseaseName": _ENFERMEDADES[rng.integers(len(_ENFERMEDADES))],
            "zona": ZONAS[rng.integers(len(ZONAS))],
            "arquetipo_real": arq,  # solo para validar; no entra al modelo
            # features con ruido gaussiano alrededor del arquetipo
            "cnnConfidence": _ruido(rng, base.get("cnnConfidence", 0.7), 0.05),
            "topKEntropy": _ruido(rng, base.get("topKEntropy", 0.8), 0.3, lo=0, hi=2.32),
            "llmConfianzaAjustada": _ruido(rng, base.get("cnnConfidence", 0.7) - 0.02, 0.05),
            "llmEstado": base.get("llmEstado", 1.0),
            "llmSinDocumentos": base.get("llmSinDocumentos", 0.0),
            "sintomasCount": max(0, round(_ruido(rng, 3, 1.5, lo=0, hi=10))),
            "avisosCount": max(0, round(_ruido(rng, base.get("avisosCount", 1.0), 0.8, lo=0, hi=5))),
            "diagnosticoLength": max(0, _ruido(rng, 500, 150, lo=0, hi=1000)),
            "completionRate": _ruido(rng, base.get("completionRate", 0.5), 0.1),
            "daysSinceCreation": max(0, round(_ruido(rng, base.get("daysSinceCreation", 15), 5, lo=0, hi=90))),
            "isComplete": 1.0 if base.get("completionRate", 0.5) >= 0.99 else 0.0,
            "hasRelapse": base.get("hasRelapse", 0.0),
            "diagnosisMonth": int(base.get("diagnosisMonth", rng.integers(1, 13))),
        }
        registros.append(reg)
    return registros


def _ruido(rng, media, sigma, lo=0.0, hi=1.0):
    return float(np.clip(rng.normal(media, sigma), lo, hi))


# ─────────────────────────────────────────────
# Carga del modelo e inferencia
# ─────────────────────────────────────────────

_modelo_cache: Optional[dict] = None


def _cargar(ruta: Path = _RUTA_MODELO) -> Optional[dict]:
    global _modelo_cache
    if _modelo_cache is None:
        if not ruta.exists():
            return None
        with open(ruta, "rb") as fh:
            _modelo_cache = pickle.load(fh)
    return _modelo_cache


def predecir_cluster(vec: dict, ruta: Path = _RUTA_MODELO) -> dict:
    """
    Asigna un cluster fitosanitario a un feature vector crudo.
    Devuelve {cluster_id, cluster_label}. Lanza si no hay modelo entrenado.
    """
    modelo = _cargar(ruta)
    if modelo is None:
        raise RuntimeError(
            "No hay modelo de clustering entrenado. "
            "Ejecuta scripts/entrenar_clustering.py primero.")
    x = normalizar(vec).reshape(1, -1)
    cid = int(modelo["kmeans"].predict(x)[0])
    return {"cluster_id": cid, "cluster_label": modelo["id2label"].get(cid, f"cluster_{cid}")}
