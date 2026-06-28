"""
evaluar_busqueda.py — Métricas del motor de búsqueda (Requisito #2).

Evalúa y COMPARA los tres métodos de recuperación sobre un conjunto de consultas
etiquetadas:
  - TF-IDF (léxico)          → almacen_documentos.buscar
  - BERT (semántico)         → busqueda_semantica.buscar_semantico
  - Híbrido (TF-IDF + BERT)  → busqueda_semantica.buscar_hibrido

Métricas (estándar en Recuperación de Información):
  - P@k  (precisión en el top-k): proporción de documentos relevantes en los k primeros.
  - MRR  (Mean Reciprocal Rank): 1/posición del primer relevante, promediado.
  - Éxito@k (hit rate): % de consultas con al menos un relevante en el top-k.

Relevancia: un documento es relevante si su CULTIVO coincide con el cultivo
esperado de la consulta (etiqueta automática a partir de los metadatos del corpus).

USO (requiere haber corrido antes construir_corpus.py):
    set PYTHONUTF8=1
    python evaluar_busqueda.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modulos.almacen_documentos import buscar
from modulos.busqueda_semantica import buscar_semantico, buscar_hibrido

# ── Conjunto de evaluación: consulta → cultivo relevante ──────────────────────
# Consultas redactadas como las escribiría un usuario (lenguaje natural, español).
_CONSULTAS = [
    ("polvo blanco como ceniza en las hojas", "calabaza"),
    ("hojas de calabaza con moho y manchas amarillas", "calabaza"),
    ("cenicilla en la planta de calabaza", "calabaza"),
    ("manchas alargadas de color café en las hojas de maíz", "maíz"),
    ("pústulas de roya en el maíz", "maíz"),
    ("tizón en las hojas del maíz", "maíz"),
    ("manchas en las vainas y hojas del frijol", "frijol"),
    ("plaga que daña la planta de frijol", "frijol"),
    ("manchas negras concéntricas en hojas de tomate", "tomate"),
    ("hojas de tomate enrolladas y amarillas por virus", "tomate"),
    ("tizón tardío en tomate", "tomate"),
    ("manchas en las hojas de la fresa", "fresa"),
    ("tizón y pudrición en el cultivo de papa", "papa"),
    ("enfermedad con manchas en el chile", "chile"),
    ("manchas y daños en hojas de limón y naranja", "cítrico"),
]

_KS = [1, 3, 5]
_TOP_K = 10


def _es_relevante(doc: dict, cultivo_esperado: str) -> bool:
    """Relevante si el documento es del cultivo esperado."""
    return doc.get("cultivo", "").lower() == cultivo_esperado.lower()


def _metricas_consulta(resultados: list[dict], cultivo: str) -> dict:
    """Calcula P@k, RR y éxito@k para una sola consulta."""
    relevancias = [_es_relevante(d, cultivo) for d in resultados]

    m = {}
    for k in _KS:
        topk = relevancias[:k]
        m[f"P@{k}"] = sum(topk) / k if k else 0.0
        m[f"exito@{k}"] = 1.0 if any(topk) else 0.0

    # Reciprocal Rank: 1 / posición del primer relevante (0 si no hay)
    rr = 0.0
    for i, rel in enumerate(relevancias, start=1):
        if rel:
            rr = 1.0 / i
            break
    m["RR"] = rr
    return m


def _evaluar_metodo(nombre: str, fn_buscar) -> dict:
    """Promedia las métricas de un método sobre todas las consultas."""
    acum = {}
    n = len(_CONSULTAS)
    for consulta, cultivo in _CONSULTAS:
        resultados = fn_buscar(consulta, top_k=_TOP_K)
        m = _metricas_consulta(resultados, cultivo)
        for clave, val in m.items():
            acum[clave] = acum.get(clave, 0.0) + val
    promedio = {clave: val / n for clave, val in acum.items()}
    promedio["_nombre"] = nombre
    return promedio


def main():
    print(f"Evaluando {len(_CONSULTAS)} consultas (top_k={_TOP_K})...\n")

    # Adaptadores para una firma común fn(consulta, top_k)
    metodos = {
        "TF-IDF": lambda c, top_k: buscar(c, top_k=top_k),
        "BERT": lambda c, top_k: buscar_semantico(c, top_k=top_k),
        "Híbrido": lambda c, top_k: buscar_hibrido(c, top_k=top_k),
    }

    resultados = [_evaluar_metodo(nombre, fn) for nombre, fn in metodos.items()]

    # Tabla comparativa
    cols = ["P@1", "P@3", "P@5", "exito@3", "exito@5", "RR"]
    encabezado = f"{'Método':<10} " + " ".join(f"{c:>8}" for c in cols)
    print(encabezado)
    print("-" * len(encabezado))
    for r in resultados:
        fila = f"{r['_nombre']:<10} " + " ".join(f"{r[c]:>8.3f}" for c in cols)
        print(fila)

    print("\nLeyenda: P@k=precisión top-k | exito@k=hit rate | RR=MRR (rango recíproco medio)")
    print("Mejor método por MRR:",
          max(resultados, key=lambda r: r["RR"])["_nombre"])

    return resultados


if __name__ == "__main__":
    main()
