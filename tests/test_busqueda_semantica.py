"""
tests/test_busqueda_semantica.py
Prueba del módulo busqueda_semantica (BERT + híbrido).
Ejecutar desde la raíz del proyecto:
    python -m tests.test_busqueda_semantica
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modulos.almacen_documentos import (
    cargar_desde_directorio,
    construir_indice,
)
from modulos.busqueda_semantica import (
    construir_embeddings,
    buscar_semantico,
    buscar_hibrido,
)

_DIR = Path(__file__).resolve().parent.parent
_BD_TEST = _DIR / "datos" / "test_bert.db"
_TFIDF_TEST = _DIR / "datos" / "test_bert_tfidf.pkl"
_EMB_TEST = _DIR / "datos" / "test_bert_embeddings.pkl"


def limpiar():
    for f in [_BD_TEST, _TFIDF_TEST, _EMB_TEST]:
        if f.exists():
            f.unlink()


def test_busqueda_semantica():
    print("\n=== Preparando datos de prueba ===")
    n = cargar_desde_directorio(ruta_bd=_BD_TEST)
    print(f"  Documentos cargados: {n}")
    construir_indice(ruta_bd=_BD_TEST, ruta_tfidf=_TFIDF_TEST)
    construir_embeddings(ruta_bd=_BD_TEST, ruta_embeddings=_EMB_TEST)

    # ── TEST 1: búsqueda semántica pura ──────────────────────────────────
    print("\n─── TEST 1: buscar_semantico('polvo blanco harinoso en las hojas') ───")
    # El objetivo: aunque la consulta no dice 'oidio' ni 'cenicilla',
    # el modelo semántico debe encontrar el documento de calabaza/oídio entre los resultados.
    resultados = buscar_semantico(
        "polvo blanco harinoso en las hojas",
        top_k=3,
        ruta_bd=_BD_TEST,
        ruta_embeddings=_EMB_TEST,
    )
    print(f"  Resultados: {len(resultados)}")
    for r in resultados:
        print(f"    score_bert={r['score_bert']:.4f}  {r['cultivo']} — {r['enfermedad']}")
    assert len(resultados) >= 1, "No se devolvió ningún resultado"
    enfermedades = [r["enfermedad"] for r in resultados]
    assert "oídio" in enfermedades, (
        f"Se esperaba que 'oídio' apareciera en los resultados, se obtuvo: {enfermedades}"
    )
    print("  OK: búsqueda semántica encontró oídio sin mencionar la palabra.")

    # ── TEST 2: búsqueda híbrida ──────────────────────────────────────────
    print("\n─── TEST 2: buscar_hibrido('manchas oscuras en hojas de tomate') ───")
    resultados_h = buscar_hibrido(
        "manchas oscuras en hojas de tomate",
        top_k=3,
        ruta_bd=_BD_TEST,
        ruta_tfidf=_TFIDF_TEST,
        ruta_embeddings=_EMB_TEST,
    )
    print(f"  Resultados: {len(resultados_h)}")
    for r in resultados_h:
        print(
            f"    tfidf={r['score_tfidf']:.4f}  bert={r['score_bert']:.4f}"
            f"  hibrido={r['score_hibrido']:.4f}  {r['cultivo']} — {r['enfermedad']}"
        )
    assert len(resultados_h) >= 1, "No se devolvió ningún resultado híbrido"
    assert resultados_h[0]["enfermedad"] == "tizón tardío", (
        f"Se esperaba 'tizón tardío' primero, se obtuvo '{resultados_h[0]['enfermedad']}'"
    )
    print("  OK: búsqueda híbrida devolvió tizón tardío del tomate primero.")

    # ── TEST 3: filtro por cultivo en híbrido ─────────────────────────────
    print("\n─── TEST 3: buscar_hibrido filtrado por cultivo=['maíz'] ───")
    resultados_maiz = buscar_hibrido(
        "lesiones en hojas",
        cultivos=["maíz"],
        top_k=5,
        ruta_bd=_BD_TEST,
        ruta_tfidf=_TFIDF_TEST,
        ruta_embeddings=_EMB_TEST,
    )
    print(f"  Resultados: {len(resultados_maiz)}")
    for r in resultados_maiz:
        print(f"    {r['cultivo']} — {r['enfermedad']}  (hibrido={r['score_hibrido']:.4f})")
        assert r["cultivo"] == "maíz", f"Filtro falló: cultivo='{r['cultivo']}'"
    print("  OK: filtro por cultivo respetado en búsqueda híbrida.")

    # ── TEST 4: score híbrido = combinación ponderada ─────────────────────
    print("\n─── TEST 4: verificar que score_hibrido = 0.4*tfidf + 0.6*bert ───")
    if resultados_h:
        r = resultados_h[0]
        esperado = round(0.4 * r["score_tfidf"] + 0.6 * r["score_bert"], 4)
        assert abs(r["score_hibrido"] - esperado) < 1e-3, (
            f"Score híbrido incorrecto: {r['score_hibrido']} != {esperado}"
        )
        print(f"  OK: {r['score_hibrido']} ≈ 0.4×{r['score_tfidf']} + 0.6×{r['score_bert']}")

    print("\nTodos los tests de Fase 2 pasaron.\n")


if __name__ == "__main__":
    limpiar()
    try:
        test_busqueda_semantica()
    finally:
        limpiar()
