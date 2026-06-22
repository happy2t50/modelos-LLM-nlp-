"""
tests/test_almacen.py
Prueba básica del módulo almacen_documentos.
Ejecutar desde la raíz del proyecto:
    python -m tests.test_almacen
"""

import sys
from pathlib import Path

# Para poder importar desde modulos/ sin instalar el paquete
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modulos.almacen_documentos import (
    cargar_desde_directorio,
    construir_indice,
    buscar,
    guardar_topk,
    recuperar_topk,
    listar_documentos,
)

# Usar BD y TF-IDF temporales para los tests
_DIR = Path(__file__).resolve().parent.parent
_BD_TEST = _DIR / "datos" / "test_almacen.db"
_TFIDF_TEST = _DIR / "datos" / "test_tfidf.pkl"


def limpiar():
    """Elimina los archivos de prueba si existen."""
    for f in [_BD_TEST, _TFIDF_TEST]:
        if f.exists():
            f.unlink()


def test_carga_y_busqueda():
    print("\n─── TEST 1: carga de documentos desde directorio ───")
    n = cargar_desde_directorio(ruta_bd=_BD_TEST)
    print(f"  Documentos insertados: {n}")
    assert n >= 3, f"Se esperaban al menos 3 documentos, se cargaron {n}"

    docs = listar_documentos(ruta_bd=_BD_TEST)
    print(f"  Total en BD: {len(docs)}")
    for d in docs:
        print(f"    [{d['id']}] {d['cultivo']} — {d['enfermedad']}")

    print("\n─── TEST 2: construcción del índice TF-IDF ───")
    construir_indice(ruta_bd=_BD_TEST, ruta_tfidf=_TFIDF_TEST)
    assert _TFIDF_TEST.exists(), "El archivo del índice TF-IDF no se creó"
    print("  Índice creado correctamente.")

    print("\n─── TEST 3: búsqueda 'tizón tardío tomate' ───")
    resultados = buscar(
        "tizón tardío tomate",
        top_k=3,
        ruta_bd=_BD_TEST,
        ruta_tfidf=_TFIDF_TEST,
    )
    print(f"  Resultados: {len(resultados)}")
    for r in resultados:
        print(f"    score={r['score']:.4f}  {r['cultivo']} — {r['enfermedad']}")
    assert len(resultados) >= 1, "Debería devolver al menos 1 resultado"
    assert resultados[0]["enfermedad"] == "tizón tardío", (
        f"Se esperaba 'tizón tardío', se obtuvo '{resultados[0]['enfermedad']}'"
    )
    print("  ✓ El primer resultado es tizón tardío del tomate.")

    print("\n─── TEST 4: filtrado por cultivo ───")
    resultados_maiz = buscar(
        "manchas hojas",
        cultivos=["maíz"],
        top_k=5,
        ruta_bd=_BD_TEST,
        ruta_tfidf=_TFIDF_TEST,
    )
    print(f"  Resultados para 'maíz': {len(resultados_maiz)}")
    for r in resultados_maiz:
        print(f"    {r['cultivo']} — {r['enfermedad']}  (score={r['score']:.4f})")
        assert r["cultivo"] == "maíz", f"Filtro falló: se coló cultivo '{r['cultivo']}'"
    print("  ✓ El filtro por cultivo funciona correctamente.")

    print("\n─── TEST 5: caché Top-K ───")
    guardar_topk("tizón tardío", resultados, ruta_bd=_BD_TEST)
    recuperados = recuperar_topk("tizón tardío", ruta_bd=_BD_TEST)
    print(f"  Documentos recuperados del caché: {len(recuperados)}")
    assert len(recuperados) == len(resultados), "El caché no guardó todos los documentos"
    print("  ✓ guardar_topk y recuperar_topk funcionan.")

    print("\n─── TEST 6: caché para enfermedad inexistente ───")
    vacio = recuperar_topk("enfermedad_que_no_existe", ruta_bd=_BD_TEST)
    assert vacio == [], f"Se esperaba lista vacía, se obtuvo: {vacio}"
    print("  ✓ Devuelve lista vacía para enfermedad sin caché.")

    print("\n✅ Todos los tests pasaron.\n")


if __name__ == "__main__":
    limpiar()
    try:
        test_carga_y_busqueda()
    finally:
        limpiar()
