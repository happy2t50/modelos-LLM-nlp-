"""
tests/test_fusion.py
Prueba básica del módulo fusion.
Ejecutar desde la raíz del proyecto:
    python -m tests.test_fusion
"""

import sys
from pathlib import Path

# Para poder importar desde modulos/ sin instalar el paquete
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modulos.fusion import combinar, diagnosticar


def test_refuerzo():
    print("\n─── TEST 1: el texto refuerza (Mildiu 0.72 + 'humedad') ───")
    cnn = {"cultivo": "tomate", "enfermedad": "mildiu", "confianza": 0.72}
    res = diagnosticar(cnn, "hojas amarillas y humedad en la planta")
    print(f"  estado: {res['estado']}")
    print(f"  confianza: {res['confianza_original']} → {res['confianza_ajustada']}")
    print(f"  refuerzo: {res['sintomas_refuerzo']}")
    print(f"  consulta: '{res['consulta']}'")
    assert res["estado"] == "reforzado"
    assert res["confianza_ajustada"] > res["confianza_original"]
    print("  ✓ Diagnóstico reforzado y confianza al alza.")


def test_contradiccion():
    print("\n─── TEST 2: el texto contradice (Mildiu 0.72 + 'polvo blanco') ───")
    cnn = {"cultivo": "tomate", "enfermedad": "mildiu", "confianza": 0.72}
    # 'polvo blanco' apunta a oídio, no a mildiu
    res = diagnosticar(cnn, "tiene un polvo blanco en las hojas")
    print(f"  estado: {res['estado']}")
    print(f"  confianza: {res['confianza_original']} → {res['confianza_ajustada']}")
    print(f"  contradicción: {res['sintomas_contradiccion']}")
    assert res["estado"] == "posible_contradiccion"
    assert res["confianza_ajustada"] < res["confianza_original"]
    print("  ✓ Confianza a la baja por contradicción.")


def test_sin_senal():
    print("\n─── TEST 3: texto sin señal sobre la enfermedad ───")
    cnn = {"cultivo": "tomate", "enfermedad": "mildiu", "confianza": 0.72}
    res = diagnosticar(cnn, "la planta está en el patio de la casa")
    print(f"  estado: {res['estado']}")
    print(f"  confianza: {res['confianza_original']} → {res['confianza_ajustada']}")
    assert res["estado"] == "sin_senal_textual"
    assert res["confianza_ajustada"] == res["confianza_original"]
    print("  ✓ Confianza sin cambios.")


def test_etiqueta_ingles():
    print("\n─── TEST 4: etiqueta CNN en inglés estilo PlantVillage ───")
    # La CNN real devuelve algo como 'Tomato___Late_blight' (tizón = blight)
    cnn = {"cultivo": "Tomato", "enfermedad": "Tomato___Late_blight", "confianza": 0.80}
    res = diagnosticar(cnn, "manchas marrones y mucha humedad")
    print(f"  estado: {res['estado']}")
    print(f"  refuerzo: {res['sintomas_refuerzo']}")
    assert res["estado"] == "reforzado", "Debería mapear 'blight' → tizón y reforzar"
    print("  ✓ Mapea etiqueta en inglés y reconoce el refuerzo.")


def test_consulta_enriquecida():
    print("\n─── TEST 5: consulta enriquecida para el buscador ───")
    cnn = {"cultivo": "tomate", "enfermedad": "mildiu", "confianza": 0.72}
    res = combinar(cnn, ["humedad", "manchas"])
    print(f"  consulta: '{res['consulta']}'")
    assert "tomate" in res["consulta"] and "mildiu" in res["consulta"]
    assert "humedad" in res["consulta"] and "manchas" in res["consulta"]
    print("  ✓ La consulta combina cultivo + enfermedad + síntomas.")


def test_confianza_acotada():
    print("\n─── TEST 6: la confianza ajustada queda acotada a 0.99 ───")
    cnn = {"cultivo": "tomate", "enfermedad": "oidio", "confianza": 0.97}
    res = combinar(cnn, ["oidio", "cenicilla", "polvo blanco"])
    print(f"  confianza: {res['confianza_original']} → {res['confianza_ajustada']}")
    assert res["confianza_ajustada"] <= 0.99
    print("  ✓ No supera 0.99.")


if __name__ == "__main__":
    test_refuerzo()
    test_contradiccion()
    test_sin_senal()
    test_etiqueta_ingles()
    test_consulta_enriquecida()
    test_confianza_acotada()
    print("\n✅ Todos los tests pasaron.\n")
