"""
app/servicio.py — Adaptador entre la API y el núcleo (modulos.asistente).

La CNN corre en el dispositivo; aquí se INYECTA su resultado en el orquestador,
que hace fusión + recuperación (RAG) + generación (LLM). La salida se mapea al
contrato LlmResponseEntity que espera la app.
"""

from modulos.asistente import consultar


def ejecutar_consulta(req) -> dict:
    """
    Ejecuta el pipeline RAG a partir del resultado de la CNN (enviado por el
    cliente) y el texto. Devuelve un dict con el contrato LlmResponse + modo.
    """
    # Resultado de la CNN provisto por el cliente (dispositivo).
    resultado_cnn = {
        "cultivo": (req.cropName or "").strip().lower(),
        "enfermedad": (req.diseaseName or "").strip().lower(),
        "confianza": float(req.confidence or 0.0),
    }

    res = consultar(
        imagen=None,                 # la CNN ya corrió en el dispositivo
        texto=req.texto or "",
        rol=req.rol,
        cultivos=req.cultivos,       # None = sin filtro; lista = parcela del usuario
        resultado_cnn=resultado_cnn,
        forzar_offline=False,        # el microservicio siempre es modo online
    )

    diag = res["diagnostico"]
    resp = res["respuesta"]

    return {
        "diagnostico": resp.get("diagnostico") or resp.get("texto", ""),
        "tratamiento": resp.get("tratamiento", ""),
        "prevencion": resp.get("prevencion", ""),
        "fuentes": resp.get("fuentes", []),
        "confianzaAjustada": diag.get("confianza_ajustada", 0.0),
        "estado": diag.get("estado", ""),
        "explicacion": diag.get("explicacion", ""),
        "sintomas": res.get("sintomas", []),
        "avisos": res.get("avisos", []),
        "sinDocumentos": resp.get("sin_documentos", False),
        "modo": res.get("modo", "online"),
    }
