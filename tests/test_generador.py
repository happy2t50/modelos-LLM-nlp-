"""
tests/test_generador.py
Prueba del módulo generador (Qwen vía Ollama).
Ejecutar desde la raíz del proyecto:
    python -m tests.test_generador

Las pruebas que no requieren Ollama corren siempre. La prueba de generación real
se salta con un aviso si el servidor Ollama no está disponible.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from modulos.generador import (
    responder,
    _construir_prompt,
    _extraer_secciones,
    _fuentes_de_documentos,
    _URL_OLLAMA,
    _MODELO,
)

# ── Datos de prueba (no dependen de la BD) ────────────────────────────────────

_DIAGNOSTICO = {
    "cultivo": "tomate",
    "enfermedad": "tizón tardío",
    "confianza_original": 0.72,
    "confianza_ajustada": 0.79,
    "estado": "reforzado",
}

_SINTOMAS = ["manchas marrones", "humedad", "moho blanco"]

_DOCUMENTOS = [
    {
        "id": 1,
        "cultivo": "tomate",
        "enfermedad": "tizón tardío",
        "fuente": "Manual Fitosanitario SAGARPA 2019",
        "texto": (
            "El tizón tardío del tomate es causado por Phytophthora infestans. "
            "Síntomas: manchas acuosas verde oscuro que se vuelven marrón-negro, "
            "moho blanco-grisáceo en el envés. Tratamiento: eliminar plantas "
            "afectadas y aplicar Mancozeb 80% PM a 2-2.5 kg/ha de forma preventiva. "
            "Prevención: evitar riego por aspersión, usar riego por goteo y rotar "
            "cultivos cada 2-3 años."
        ),
    },
]


def _ollama_disponible() -> bool:
    """Comprueba si el servidor Ollama responde."""
    try:
        base = _URL_OLLAMA.rsplit("/api/", 1)[0]
        r = requests.get(base + "/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ── TEST 1: parseo de secciones (sin Ollama) ─────────────────────────────────

def test_extraer_secciones():
    print("\n─── TEST 1: _extraer_secciones ───")
    texto = (
        "DIAGNÓSTICO:\nEl tomate tiene tizón tardío.\n\n"
        "TRATAMIENTO:\nAplicar Mancozeb 2 kg/ha.\n\n"
        "PREVENCIÓN:\nRiego por goteo.\n\n"
        "FUENTES:\nManual SAGARPA 2019."
    )
    sec = _extraer_secciones(texto)
    print(f"  diagnostico: {sec['diagnostico']!r}")
    print(f"  tratamiento: {sec['tratamiento']!r}")
    print(f"  prevencion:  {sec['prevencion']!r}")
    assert "tizón tardío" in sec["diagnostico"]
    assert "Mancozeb" in sec["tratamiento"]
    assert "goteo" in sec["prevencion"]
    assert "SAGARPA" in sec["fuentes"]
    print("  OK: las cuatro secciones se separaron correctamente.")


# ── TEST 2: fuentes derivadas de los documentos (sin Ollama) ─────────────────

def test_fuentes():
    print("\n─── TEST 2: _fuentes_de_documentos ───")
    docs = _DOCUMENTOS + [
        {"fuente": "Manual Fitosanitario SAGARPA 2019"},  # duplicada
        {"fuente": "INIFAP 2020"},
        {"fuente": ""},  # vacía, se ignora
    ]
    fuentes = _fuentes_de_documentos(docs)
    print(f"  fuentes: {fuentes}")
    assert fuentes == ["Manual Fitosanitario SAGARPA 2019", "INIFAP 2020"], fuentes
    print("  OK: fuentes únicas, sin duplicados ni vacías.")


# ── TEST 3: el prompt contiene las reglas de seguridad (sin Ollama) ──────────

def test_prompt_incluye_reglas():
    print("\n─── TEST 3: el prompt incluye reglas e info ───")
    prompt = _construir_prompt(_DIAGNOSTICO, _SINTOMAS, _DOCUMENTOS, "aprendiz")
    assert "ÚNICAMENTE con información contenida en los DOCUMENTOS" in prompt
    assert "NUNCA inventes dosis" in prompt
    assert "tizón tardío" in prompt
    assert "APRENDIZ" in prompt  # estilo del rol
    print("  OK: el prompt incluye reglas, diagnóstico y estilo del rol.")


# ── TEST 4: respuesta sin documentos (sin Ollama) ────────────────────────────

def test_responder_sin_documentos():
    print("\n─── TEST 4: responder() sin documentos ───")
    r = responder(_DIAGNOSTICO, _SINTOMAS, [], rol="agricultor")
    print(f"  texto: {r['texto']}")
    assert r["sin_documentos"] is True
    assert "agrónomo" in r["texto"].lower()
    assert r["fuentes"] == []
    print("  OK: sin documentos avisa y no inventa.")


# ── TEST 5: generación real con Qwen (requiere Ollama) ───────────────────────

def test_generacion_real():
    print("\n─── TEST 5: generación real con Qwen (agricultor vs aprendiz) ───")
    if not _ollama_disponible():
        print("  ⚠ Ollama no disponible; se omite la prueba de generación real.")
        return

    print(f"  Usando modelo: {_MODELO}")

    print("\n  >>> ROL: agricultor")
    r_agri = responder(_DIAGNOSTICO, _SINTOMAS, _DOCUMENTOS, rol="agricultor")
    print(r_agri["texto"])
    assert r_agri["texto"].strip(), "Respuesta vacía (agricultor)"
    assert r_agri["fuentes"], "No se derivaron fuentes"

    print("\n  >>> ROL: aprendiz")
    r_apr = responder(_DIAGNOSTICO, _SINTOMAS, _DOCUMENTOS, rol="aprendiz")
    print(r_apr["texto"])
    assert r_apr["texto"].strip(), "Respuesta vacía (aprendiz)"

    # Heurística suave: la respuesta de aprendiz suele ser más larga/técnica.
    print(f"\n  longitud agricultor={len(r_agri['texto'])}  "
          f"aprendiz={len(r_apr['texto'])}")
    print("  OK: generó respuesta para ambos roles.")


if __name__ == "__main__":
    test_extraer_secciones()
    test_fuentes()
    test_prompt_incluye_reglas()
    test_responder_sin_documentos()
    test_generacion_real()
    print("\nPruebas de Fase 6 completadas.\n")
