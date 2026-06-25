"""
tests/test_asistente.py
Prueba del orquestador (Fase 7): conexión, modos online/offline y caché Top-K.
Ejecutar desde la raíz del proyecto:
    python -m tests.test_asistente

Usa un generador FALSO inyectado para no depender de Ollama. Construye un almacén
temporal propio (BD, TF-IDF y embeddings) para no tocar los datos reales.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modulos import mis_cultivos, conexion
from modulos.almacen_documentos import (
    cargar_desde_directorio,
    construir_indice,
    recuperar_topk,
)
from modulos.busqueda_semantica import construir_embeddings
from modulos.asistente import consultar, precargar_cache

_DIR = Path(__file__).resolve().parent.parent
_BD = _DIR / "datos" / "test_asistente.db"
_TFIDF = _DIR / "datos" / "test_asistente_tfidf.pkl"
_EMB = _DIR / "datos" / "test_asistente_emb.pkl"


def limpiar():
    for f in [_BD, _TFIDF, _EMB]:
        if f.exists():
            f.unlink()


def _generador_falso(diagnostico, sintomas, documentos, rol):
    """Generador de prueba: no llama a Ollama; solo confirma qué recibió."""
    return {
        "texto": f"[FALSO] rol={rol}, {len(documentos)} documentos",
        "diagnostico": diagnostico.get("enfermedad", ""),
        "tratamiento": "",
        "prevencion": "",
        "fuentes": [d.get("fuente", "") for d in documentos],
        "rol": rol,
        "sin_documentos": not documentos,
    }


# Resultado de CNN inyectado (la CNN real llega en Fase 8)
_CNN_TOMATE = {"cultivo": "tomate", "enfermedad": "tizón tardío", "confianza": 0.72}


def preparar_almacen():
    print("=== Preparando almacén temporal ===")
    cargar_desde_directorio(ruta_bd=_BD)
    construir_indice(ruta_bd=_BD, ruta_tfidf=_TFIDF)
    construir_embeddings(ruta_bd=_BD, ruta_embeddings=_EMB)
    # Registrar cultivos de la parcela
    mis_cultivos.limpiar_todo(ruta_bd=_BD)
    for c in ["tomate", "calabaza", "maíz"]:
        mis_cultivos.agregar(c, ruta_bd=_BD)
    print(f"  cultivos: {mis_cultivos.listar(ruta_bd=_BD)}")


def test_conexion():
    print("\n─── TEST 1: detección de conexión ───")
    estado = conexion.estado_conexion()
    print(f"  estado real de la red: {estado}")
    assert estado in ("online", "offline")
    assert isinstance(conexion.hay_internet(), bool)
    print("  OK: hay_internet() devuelve un booleano.")


def test_precarga_cache():
    print("\n─── TEST 2: precargar_cache (caché semilla) ───")
    res = precargar_cache(
        forzar_online=True, ruta_bd=_BD, ruta_tfidf=_TFIDF, ruta_embeddings=_EMB,
    )
    print(f"  resultado: ok={res['ok']}, total={res['total']}")
    print(f"  enfermedades cacheadas: {res['enfermedades_cacheadas']}")
    assert res["ok"], res["motivo"]
    assert "tizón tardío" in res["enfermedades_cacheadas"]
    # Verificar que quedó guardado en el caché
    cacheado = recuperar_topk("tizón tardío", ruta_bd=_BD)
    assert len(cacheado) >= 1, "El caché Top-K no se guardó"
    print(f"  OK: 'tizón tardío' quedó en caché con {len(cacheado)} documento(s).")


def test_consulta_online():
    print("\n─── TEST 3: consultar() en modo ONLINE ───")
    res = consultar(
        imagen=None,
        texto="manchas marrones y mucha humedad en las hojas",
        rol="agricultor",
        resultado_cnn=_CNN_TOMATE,
        forzar_offline=False,        # forzar online
        fn_generar=_generador_falso,
        ruta_bd=_BD, ruta_tfidf=_TFIDF, ruta_embeddings=_EMB,
    )
    print(f"  modo: {res['modo']}")
    print(f"  cultivos: {res['cultivos']}")
    print(f"  n_documentos: {res['n_documentos']}")
    print(f"  respuesta: {res['respuesta']['texto']}")
    assert res["modo"] == "online"
    assert res["n_documentos"] >= 1
    # En online se debe haber (re)guardado el Top-K
    assert recuperar_topk("tizón tardío", ruta_bd=_BD), "Online no actualizó el caché"
    print("  OK: modo online recuperó documentos y actualizó el caché.")


def test_consulta_offline():
    print("\n─── TEST 4: consultar() en modo OFFLINE (desde caché) ───")
    res = consultar(
        imagen=None,
        texto="manchas marrones y mucha humedad en las hojas",
        rol="aprendiz",
        resultado_cnn=_CNN_TOMATE,
        forzar_offline=True,         # forzar offline
        fn_generar=_generador_falso,
        ruta_bd=_BD, ruta_tfidf=_TFIDF, ruta_embeddings=_EMB,
    )
    print(f"  modo: {res['modo']}")
    print(f"  n_documentos: {res['n_documentos']}")
    for d in res["documentos"]:
        print(f"    {d.get('cultivo')} — {d.get('enfermedad')}")
    assert res["modo"] == "offline"
    assert res["n_documentos"] >= 1, "Offline no recuperó nada del caché"
    # El documento de la enfermedad diagnosticada debe estar entre los del caché.
    enfermedades = [d.get("enfermedad") for d in res["documentos"]]
    assert "tizón tardío" in enfermedades, (
        f"El caché offline no contiene la enfermedad diagnosticada: {enfermedades}"
    )
    print("  OK: modo offline respondió desde el caché Top-K (contiene la enfermedad).")


def test_offline_filtra_por_cultivo():
    print("\n─── TEST 5: offline respeta el filtro de 'mis cultivos' ───")
    # Quitar tomate de la parcela: la consulta de tomate ya no debe colar ese doc.
    mis_cultivos.quitar("tomate", ruta_bd=_BD)
    print(f"  cultivos ahora: {mis_cultivos.listar(ruta_bd=_BD)}")
    res = consultar(
        imagen=None,
        texto="manchas marrones y humedad",
        rol="agricultor",
        resultado_cnn=_CNN_TOMATE,
        forzar_offline=True,
        fn_generar=_generador_falso,
        ruta_bd=_BD, ruta_tfidf=_TFIDF, ruta_embeddings=_EMB,
    )
    print(f"  n_documentos: {res['n_documentos']}")
    for d in res["documentos"]:
        print(f"    {d.get('cultivo')} — {d.get('enfermedad')}")
        assert d.get("cultivo", "").lower() != "tomate", "Coló tomate pese al filtro"
    print("  OK: el filtro por cultivo se respeta también offline.")
    # Restaurar
    mis_cultivos.agregar("tomate", ruta_bd=_BD)


if __name__ == "__main__":
    limpiar()
    try:
        preparar_almacen()
        test_conexion()
        test_precarga_cache()
        test_consulta_online()
        test_consulta_offline()
        test_offline_filtra_por_cultivo()
        print("\nTodos los tests de Fase 7 pasaron.\n")
    finally:
        limpiar()
