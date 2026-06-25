"""
asistente.py
Orquestador del sistema: une CNN+NLP (diagnóstico), recuperación de documentos
(TF-IDF + BERT), caché Top-K y generación con Qwen, con lógica online/offline.

Flujo (según el plan, Fase 7):
  - Lee los cultivos de 'mis_cultivos' (filtro de la parcela).
  - Extrae síntomas del texto (NLP) y los fusiona con la predicción de la CNN.
  - CON internet  → busca fresco (filtrado por cultivos), GUARDA el TOP-K en
                    SQLite (caché) y responde. Modo 'online'.
  - SIN internet  → usa el caché Top-K de esa enfermedad; si no hay, recurre a la
                    búsqueda híbrida local (TF-IDF + BERT). Modo 'offline'.
  - precargar_cache(): con internet, descarga y guarda el TOP-K de las
                    enfermedades de los cultivos registrados (caché semilla).

Nota Fase 7: la CNN real se conecta en la Fase 8. Aquí el resultado de la CNN se
recibe como parámetro (`resultado_cnn`) o se obtiene de un stub inyectable, para
poder probar la lógica online/offline de forma aislada.
"""

from typing import Callable, Optional

from modulos import mis_cultivos, conexion, fusion
from modulos.nlp_texto import extraer_sintomas
from modulos.almacen_documentos import (
    buscar,
    guardar_topk,
    recuperar_topk,
    listar_documentos,
    _RUTA_BD,
    _RUTA_TFIDF,
)
from modulos.busqueda_semantica import buscar_hibrido, _RUTA_EMBEDDINGS
from modulos import generador

_TOP_K = 10


# ─────────────────────────────────────────────
# Stub de la CNN (se reemplaza en la Fase 8)
# ─────────────────────────────────────────────

def _clasificar_stub(imagen) -> dict:
    """
    Marcador de posición de la CNN hasta la Fase 8. Devuelve un diagnóstico
    vacío de baja confianza. En la Fase 8, clasificador.predecir() ocupa su lugar.
    """
    return {"cultivo": "", "enfermedad": "", "confianza": 0.0}


# ─────────────────────────────────────────────
# Recuperación de documentos según el modo
# ─────────────────────────────────────────────

def _buscar_online(consulta, cultivos, enfermedad, top_k,
                   ruta_bd, ruta_tfidf, ruta_embeddings) -> list[dict]:
    """
    Modo online: busca 'fresco' en el almacén local (híbrido TF-IDF + BERT),
    filtrado por cultivos, y GUARDA el Top-K en el caché para el modo offline.

    Nota: hoy la fuente de documentos es local (ver CLAUDE.md). Cuando exista una
    fuente remota (Google Drive), la búsqueda fresca consultaría esa fuente; el
    mecanismo de guardado del Top-K no cambia.
    """
    documentos = buscar_hibrido(
        consulta, cultivos=cultivos, top_k=top_k,
        ruta_bd=ruta_bd, ruta_tfidf=ruta_tfidf, ruta_embeddings=ruta_embeddings,
    )
    if documentos and enfermedad:
        guardar_topk(enfermedad, documentos, ruta_bd=ruta_bd)
    return documentos


def _buscar_offline(consulta, cultivos, enfermedad, top_k,
                    ruta_bd, ruta_tfidf, ruta_embeddings) -> list[dict]:
    """
    Modo offline: primero intenta el caché Top-K de esa enfermedad; si está vacío,
    recurre a la búsqueda híbrida local sobre el almacén.
    """
    cacheados = recuperar_topk(enfermedad, ruta_bd=ruta_bd) if enfermedad else []
    if cacheados:
        # Respeta el filtro por cultivos también sobre lo cacheado.
        if cultivos:
            cl = [c.lower() for c in cultivos]
            cacheados = [d for d in cacheados if d.get("cultivo", "").lower() in cl]
        if cacheados:
            return cacheados[:top_k]

    # Sin caché útil: búsqueda híbrida local (TF-IDF + BERT funcionan offline).
    return buscar_hibrido(
        consulta, cultivos=cultivos, top_k=top_k,
        ruta_bd=ruta_bd, ruta_tfidf=ruta_tfidf, ruta_embeddings=ruta_embeddings,
    )


# ─────────────────────────────────────────────
# Consulta principal
# ─────────────────────────────────────────────

def consultar(
    imagen,
    texto: str,
    rol: str = "agricultor",
    resultado_cnn: Optional[dict] = None,
    forzar_offline: Optional[bool] = None,
    fn_generar: Optional[Callable] = None,
    top_k: int = _TOP_K,
    ruta_bd=_RUTA_BD,
    ruta_tfidf=_RUTA_TFIDF,
    ruta_embeddings=_RUTA_EMBEDDINGS,
) -> dict:
    """
    Atiende una consulta completa (imagen + texto) y devuelve la respuesta.

    Args:
        imagen:         Ruta/objeto de la imagen (se usa con la CNN; en Fase 7 es
                        opcional si se inyecta resultado_cnn).
        texto:          Texto de síntomas del usuario.
        rol:            'agricultor' o 'aprendiz'.
        resultado_cnn:  Diagnóstico de la CNN ya calculado (Fase 8 lo provee la
                        CNN real); si es None se usa el stub.
        forzar_offline: None = detectar internet; True = forzar offline;
                        False = forzar online (útil para pruebas).
        fn_generar:     Función generadora a usar (por defecto generador.responder).
                        Inyectable para pruebas sin Ollama.
        top_k:          Número de documentos a recuperar/cachear.

    Returns:
        dict con: modo ('online'/'offline'), cultivos, diagnostico (fusión),
        sintomas, n_documentos, respuesta (salida del generador).
    """
    if fn_generar is None:
        fn_generar = generador.responder

    # 1) Cultivos de la parcela (filtro)
    cultivos = mis_cultivos.listar(ruta_bd=ruta_bd)

    # 2) NLP del texto
    sintomas = extraer_sintomas(texto)

    # 3) Diagnóstico CNN + fusión
    if resultado_cnn is None:
        resultado_cnn = _clasificar_stub(imagen)
    diagnostico = fusion.combinar(resultado_cnn, sintomas)
    consulta_ir = diagnostico["consulta"]
    enfermedad = diagnostico["enfermedad"]

    # 4) Decidir modo
    if forzar_offline is None:
        online = conexion.hay_internet()
    else:
        online = not forzar_offline
    modo = "online" if online else "offline"

    # 5) Recuperar documentos según el modo
    if online:
        documentos = _buscar_online(
            consulta_ir, cultivos, enfermedad, top_k,
            ruta_bd, ruta_tfidf, ruta_embeddings,
        )
    else:
        documentos = _buscar_offline(
            consulta_ir, cultivos, enfermedad, top_k,
            ruta_bd, ruta_tfidf, ruta_embeddings,
        )

    # 6) Generar respuesta
    respuesta = fn_generar(diagnostico, sintomas, documentos, rol)

    return {
        "modo": modo,
        "cultivos": cultivos,
        "diagnostico": diagnostico,
        "sintomas": sintomas,
        "n_documentos": len(documentos),
        "documentos": documentos,
        "respuesta": respuesta,
    }


# ─────────────────────────────────────────────
# Pre-carga del caché (caché semilla)
# ─────────────────────────────────────────────

def precargar_cache(
    forzar_online: Optional[bool] = None,
    top_k: int = _TOP_K,
    ruta_bd=_RUTA_BD,
    ruta_tfidf=_RUTA_TFIDF,
    ruta_embeddings=_RUTA_EMBEDDINGS,
) -> dict:
    """
    Descarga y guarda el Top-K de las enfermedades de los cultivos registrados,
    para que el modo offline funcione desde el día uno (caché semilla).

    Requiere internet (salvo que se fuerce con forzar_online=True en pruebas).

    Returns:
        dict con: ok (bool), motivo (str si falló), enfermedades_cacheadas (lista),
        total (int).
    """
    online = (conexion.hay_internet() if forzar_online is None else forzar_online)
    if not online:
        return {
            "ok": False,
            "motivo": "Sin internet: no se puede pre-cargar el caché.",
            "enfermedades_cacheadas": [],
            "total": 0,
        }

    cultivos = set(mis_cultivos.listar(ruta_bd=ruta_bd))
    if not cultivos:
        return {
            "ok": False,
            "motivo": "No hay cultivos registrados en 'mis_cultivos'.",
            "enfermedades_cacheadas": [],
            "total": 0,
        }

    # Enumera las (cultivo, enfermedad) presentes en el almacén para esos cultivos.
    pares = set()
    for doc in listar_documentos(ruta_bd=ruta_bd):
        if doc["cultivo"].lower() in {c.lower() for c in cultivos}:
            pares.add((doc["cultivo"], doc["enfermedad"]))

    cacheadas = []
    for cultivo, enfermedad in sorted(pares):
        consulta = f"{cultivo} {enfermedad}"
        documentos = buscar_hibrido(
            consulta, cultivos=[cultivo], top_k=top_k,
            ruta_bd=ruta_bd, ruta_tfidf=ruta_tfidf, ruta_embeddings=ruta_embeddings,
        )
        if documentos:
            guardar_topk(enfermedad, documentos, ruta_bd=ruta_bd)
            cacheadas.append(enfermedad)

    return {
        "ok": True,
        "motivo": "",
        "enfermedades_cacheadas": cacheadas,
        "total": len(cacheadas),
    }
