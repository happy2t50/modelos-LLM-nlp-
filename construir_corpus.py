"""
construir_corpus.py — Integra los PDFs nuevos con el corpus del compañero.

Qué hace:
  1. Procesa los PDFs nuevos (extrae texto, lo trocea en fragmentos y lo limpia).
  2. Carga el corpus del compañero (datos/corpus_procesado_lab1.json) y le asigna
     un cultivo a cada registro según su documento de origen.
  3. Une ambos en un corpus combinado → datos/corpus_combinado.json.
  4. Lo carga en el almacén SQLite (troceado) y reconstruye TF-IDF + embeddings,
     de modo que el sistema (ejecutar.py / asistente.py) responda con TODO el corpus.

USO:
    python construir_corpus.py
"""

import re
import json
from pathlib import Path

from pypdf import PdfReader

from modulos.almacen_documentos import (
    agregar_corpus, cargar_desde_directorio, construir_indice, listar_documentos,
    _RUTA_BD, _RUTA_TFIDF,
)
from modulos.busqueda_semantica import construir_embeddings, _RUTA_EMBEDDINGS
from modulos import mis_cultivos

_DIR = Path(__file__).resolve().parent
_DIR_PDFS = Path(r"C:/Users/umina/OneDrive/Escritorio/documentos_fitosanitarios")
_RUTA_JSON_COMPANERO = _DIR / "datos" / "corpus_procesado_lab1.json"
_RUTA_JSON_COMBINADO = _DIR / "datos" / "corpus_combinado.json"

_TAM_FRAGMENTO = 1000   # caracteres por trozo (aprox.)

# ── Manifiesto: qué PDFs integrar y con qué metadatos ─────────────────────────
# (cultivo en español/minúsculas para casar con mis_cultivos y la CNN)
_MANIFIESTO = {
    "enfermedades_calabaza_cucurbitaceas.pdf":
        ("calabaza", "enfermedades y plagas", "Guía GIP Cucurbitáceas — MAPA España 2023"),
    "enfermedades_calabaza_upr.pdf":
        ("calabaza", "enfermedades", "Producción de Calabaza — UPR Mayagüez"),
    "enfermedades_maiz_mapa.pdf":
        ("maíz", "plagas y enfermedades", "Guía GIP Maíz — MAPA España"),
    "enfermedades_tomate.pdf":
        ("tomate", "enfermedades", "Enfermedades de Tomate — INTA Ediciones"),
    "manual_bpa_tomate.pdf":
        ("tomate", "buenas prácticas", "Manual BPA Tomate — MAGyP Argentina"),
    "fitoplasmas_fresa.pdf":
        ("fresa", "fitoplasmas", "Ficha Fitoplasmas Fresa — SENASICA"),
    "gestion_plagas_fresa.pdf":
        ("fresa", "plagas", "Guía GIP Fresa y Fresón — MAPA España 2019"),
    "INIFAP_Enfermedades_Chile.pdf":
        ("chile", "enfermedades", "Enfermedades del Chile — INIFAP"),
    "manual_bpa_general.pdf":
        ("general", "buenas prácticas", "Manual BPA Frutas y Hortalizas"),
    "plagas_frijol.pdf":
        ("frijol", "plagas y enfermedades", "Plagas del Frijol — SAGARPA"),
    "plagas_papa.pdf":
        ("papa", "plagas", "Guía Plagas de Papa — INIAP Ecuador"),
    "produccion_citricos.pdf":
        ("cítrico", "producción y enfermedades", "Producción de Limón y Naranja — INIFAP Morelos"),
}

# Cultivo para los registros del compañero, según su documento de origen.
# 'general' = no se filtra por un cultivo concreto (no estorba en búsquedas filtradas).
_CULTIVO_JSON = {
    "CIMMYT_Enfermedades_Maiz_Guia_Campo": "maíz",
    "SENASICA_Manual_Vigilancia_Fitosanitaria": "general",
    "PlantVillage_paper_Hughes_Salathe_2015": "general",
    "MobileNetV2_paper_Sandler_2018": "general",
}

# Stopwords en español (para texto_limpio). Intenta NLTK; si no, set básico.
try:
    from nltk.corpus import stopwords
    _STOP = set(stopwords.words("spanish"))
except Exception:
    try:
        import nltk
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        _STOP = set(stopwords.words("spanish"))
    except Exception:
        from modulos.nlp_texto import _STOPWORDS as _STOP


def limpiar(texto: str) -> str:
    """Minúsculas, sin saltos ni puntuación, sin stopwords (estilo del Lab del equipo)."""
    texto = texto.lower()
    texto = re.sub(r"\s+", " ", texto)
    texto = re.sub(r"[^\w\sáéíóúñü]", " ", texto)
    palabras = [p for p in texto.split() if p not in _STOP and len(p) > 2]
    return " ".join(palabras)


def trocear(texto: str, tam: int = _TAM_FRAGMENTO) -> list[str]:
    """Parte el texto en fragmentos de ~tam caracteres respetando párrafos/frases."""
    texto = re.sub(r"\s+", " ", texto).strip()
    if not texto:
        return []
    # Cortar por frases y reagrupar hasta ~tam
    frases = re.split(r"(?<=[\.\!\?])\s+", texto)
    fragmentos, actual = [], ""
    for f in frases:
        if len(actual) + len(f) + 1 <= tam:
            actual = (actual + " " + f).strip()
        else:
            if actual:
                fragmentos.append(actual)
            actual = f
    if actual:
        fragmentos.append(actual)
    return [t for t in fragmentos if len(t) > 60]  # descartar trozos minúsculos


def procesar_pdfs() -> list[dict]:
    """Extrae, trocea y limpia los PDFs del manifiesto."""
    registros = []
    print("== Procesando PDFs nuevos ==")
    for archivo, (cultivo, tema, fuente) in _MANIFIESTO.items():
        ruta = _DIR_PDFS / archivo
        if not ruta.exists():
            print(f"  ⚠ no encontrado, se omite: {archivo}")
            continue
        try:
            reader = PdfReader(str(ruta))
            texto = " ".join((pg.extract_text() or "") for pg in reader.pages)
        except Exception as e:
            print(f"  ⚠ error leyendo {archivo}: {e}")
            continue

        trozos = trocear(texto)
        for t in trozos:
            registros.append({
                "documento_origen": archivo.replace(".pdf", ""),
                "tema_plaga": tema,
                "contenido_texto": t,
                "total_imagenes": 0,
                "imagenes_documento_resumen": "",
                "texto_limpio": limpiar(t),
                "cultivo": cultivo,
                "fuente": fuente,
            })
        print(f"  {archivo:42s} → {len(trozos):4d} fragmentos ({cultivo})")
    return registros


def cargar_corpus_companero() -> list[dict]:
    """Carga el JSON del compañero y le asigna cultivo + fuente."""
    print("== Cargando corpus del compañero ==")
    if not _RUTA_JSON_COMPANERO.exists():
        print(f"  ⚠ no existe {_RUTA_JSON_COMPANERO.name}; se omite.")
        return []
    data = json.load(open(_RUTA_JSON_COMPANERO, encoding="utf-8"))
    for r in data:
        origen = r.get("documento_origen", "")
        r["cultivo"] = _CULTIVO_JSON.get(origen, "general")
        r["fuente"] = origen.replace("_", " ")
    print(f"  {len(data)} registros (del Lab del equipo)")
    return data


def main():
    nuevos = procesar_pdfs()
    companero = cargar_corpus_companero()
    combinado = companero + nuevos

    # 1) Guardar corpus combinado (lo que pediste ver)
    _RUTA_JSON_COMBINADO.parent.mkdir(parents=True, exist_ok=True)
    json.dump(combinado, open(_RUTA_JSON_COMBINADO, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n== Corpus combinado: {len(combinado)} registros "
          f"({len(companero)} del equipo + {len(nuevos)} de PDFs nuevos)")
    print(f"   Guardado en {_RUTA_JSON_COMBINADO.relative_to(_DIR)}")

    # 2) Cargar al almacén SQLite (reconstruido desde cero para esquema limpio)
    if _RUTA_BD.exists():
        _RUTA_BD.unlink()
    registros_bd = [
        {"cultivo": r["cultivo"], "enfermedad": r.get("tema_plaga", ""),
         "fuente": r.get("fuente", ""), "texto": r["contenido_texto"]}
        for r in combinado if r.get("contenido_texto", "").strip()
    ]
    n = agregar_corpus(registros_bd)
    # Además, los documentos .txt curados de documentos/ (tienen dosis concretas,
    # cosa que las guías GIP no traen). Son cortos y de alta calidad.
    n_txt = cargar_desde_directorio()
    print(f"\n== Almacén: {n} fragmentos del corpus + {n_txt} documentos curados (.txt) ==")
    from collections import Counter
    docs = listar_documentos()
    print("   Por cultivo:", dict(Counter(d["cultivo"] for d in docs)))

    # 3) Reconstruir índices
    print("\n== Reconstruyendo índices ==")
    construir_indice()
    construir_embeddings()

    # 4) Registrar cultivos de la parcela
    mis_cultivos.limpiar_todo()
    for c in ["maíz", "calabaza", "frijol"]:
        mis_cultivos.agregar(c)
    print(f"\nCultivos registrados: {mis_cultivos.listar()}")
    print("\n== Corpus combinado listo. Ya puedes usar ejecutar.py ==")


if __name__ == "__main__":
    main()
