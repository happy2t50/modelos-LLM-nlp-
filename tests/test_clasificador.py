"""
tests/test_clasificador.py
Prueba del clasificador CNN (Fase 8) y de los avisos del orquestador.
Ejecutar desde la raíz del proyecto:
    python -m tests.test_clasificador

La prueba de predicción carga el modelo real (best.pth). Como no hay imágenes de
hoja en el repo, se usa una imagen sintética: NO valida la exactitud del modelo,
solo que el pipeline (carga + preprocesado + forward + parseo) funciona de punta a
punta y devuelve una clase válida.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from modulos.clasificador import predecir, _separar_clase, _CULTIVO_ES
from modulos.asistente import _avisos_imagen

_DIR = Path(__file__).resolve().parent.parent
_RUTA_PESOS = _DIR / "modelos" / "best.pth"


# ── TEST 1: parseo de etiquetas (sin modelo) ─────────────────────────────────

def test_separar_clase():
    print("\n─── TEST 1: _separar_clase (los 3 formatos) ───")
    casos = {
        "Tomato___Late_blight": ("tomate", "late blight"),
        "Corn_(maize)___Northern_Leaf_Blight": ("maíz", "northern leaf blight"),
        "Calabaza_Downy Mildew": ("calabaza", "downy mildew"),
        "Calabaza_Powdery_Mildew": ("calabaza", "powdery mildew"),
        "Black spot": ("cítrico", "black spot"),
        "Tomato___healthy": ("tomate", "sano"),
        "Calabaza_Healthy Leaf": ("calabaza", "sano"),
        "Frijol_Buenos": ("frijol", "sano"),
    }
    for clase, esperado in casos.items():
        obtenido = _separar_clase(clase)
        print(f"  {clase!r:42s} → {obtenido}")
        assert obtenido == esperado, f"{clase}: esperado {esperado}, obtenido {obtenido}"
    print("  OK: los tres formatos de etiqueta se parsean bien.")


# ── TEST 2: predicción real sobre imagen sintética ───────────────────────────

def test_predecir_pipeline():
    print("\n─── TEST 2: predecir() de punta a punta ───")
    if not _RUTA_PESOS.exists():
        print(f"  ⚠ No se encontró {_RUTA_PESOS.name}; se omite.")
        return

    # Imagen sintética (verde, simulando follaje). No esperamos acierto, solo pipeline.
    img = Image.new("RGB", (400, 400), color=(60, 140, 60))
    res = predecir(img)
    print(f"  cultivo:    {res['cultivo']}")
    print(f"  enfermedad: {res['enfermedad']}")
    print(f"  confianza:  {res['confianza']}")
    print(f"  clase_cnn:  {res['clase_cnn']}")
    print(f"  confianza_baja: {res['confianza_baja']}")

    assert 0.0 <= res["confianza"] <= 1.0, "Confianza fuera de [0,1]"
    assert res["clase_cnn"], "No devolvió etiqueta de clase"
    assert isinstance(res["cultivo"], str) and isinstance(res["enfermedad"], str)
    # El cultivo debe ser uno de los conocidos (o cítrico)
    cultivos_validos = set(_CULTIVO_ES.values()) | {"cítrico"}
    assert res["cultivo"] in cultivos_validos, f"Cultivo desconocido: {res['cultivo']}"
    print("  OK: el pipeline de la CNN corre y devuelve una clase válida.")


# ── TEST 3: avisos del orquestador (sin modelo) ──────────────────────────────

def test_avisos_confianza_baja():
    print("\n─── TEST 3: aviso por confianza baja ───")
    cnn = {"cultivo": "tomate", "enfermedad": "tizón tardío",
           "confianza": 0.31, "confianza_baja": True}
    diag = {"cultivo": "tomate", "enfermedad": "tizón tardío"}
    avisos = _avisos_imagen(cnn, diag, cultivos=["tomate"])
    for a in avisos:
        print(f"  • {a}")
    assert any("confianza" in a.lower() for a in avisos), "Falta aviso de confianza baja"
    print("  OK: avisa cuando la confianza es baja.")


def test_avisos_cultivo_fuera():
    print("\n─── TEST 4: aviso por cultivo fuera de 'mis cultivos' ───")
    cnn = {"cultivo": "manzana", "enfermedad": "apple scab",
           "confianza": 0.95, "confianza_baja": False}
    diag = {"cultivo": "manzana", "enfermedad": "apple scab"}
    avisos = _avisos_imagen(cnn, diag, cultivos=["tomate", "calabaza", "maíz"])
    for a in avisos:
        print(f"  • {a}")
    assert any("no está en tus cultivos" in a for a in avisos), "Falta aviso de cultivo fuera"
    # Confianza alta: NO debe avisar de confianza
    assert not any("confianza" in a.lower() for a in avisos), "No debería avisar de confianza"
    print("  OK: avisa cuando el cultivo no está registrado.")


def test_sin_avisos():
    print("\n─── TEST 5: sin avisos cuando todo está bien ───")
    cnn = {"cultivo": "maíz", "enfermedad": "northern leaf blight",
           "confianza": 0.88, "confianza_baja": False}
    diag = {"cultivo": "maíz", "enfermedad": "northern leaf blight"}
    avisos = _avisos_imagen(cnn, diag, cultivos=["maíz", "calabaza"])
    print(f"  avisos: {avisos}")
    assert avisos == [], "No debería haber avisos"
    print("  OK: sin avisos con confianza alta y cultivo registrado.")


if __name__ == "__main__":
    test_separar_clase()
    test_predecir_pipeline()
    test_avisos_confianza_baja()
    test_avisos_cultivo_fuera()
    test_sin_avisos()
    print("\nTodos los tests de Fase 8 pasaron.\n")
