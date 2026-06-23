"""
tests/test_nlp_texto.py
Prueba básica del módulo nlp_texto.
Ejecutar desde la raíz del proyecto:
    python -m tests.test_nlp_texto
"""

import sys
from pathlib import Path

# Para poder importar desde modulos/ sin instalar el paquete
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modulos.nlp_texto import (
    limpiar,
    tokenizar,
    extraer_sintomas,
    enriquecer_consulta,
)


def test_limpieza():
    print("\n─── TEST 1: limpieza (minúsculas, sin signos ni acentos) ───")
    salida = limpiar("¡Hojas AMARILLAS y polvo blanco!! (en el tomate #3)")
    print(f"  '{salida}'")
    assert salida == "hojas amarillas y polvo blanco en el tomate"
    print("  ✓ Texto limpiado correctamente.")


def test_tokenizar():
    print("\n─── TEST 2: tokenización sin stopwords ───")
    tokens = tokenizar("las hojas de la planta están muy amarillas")
    print(f"  {tokens}")
    # 'las', 'de', 'la', 'estan', 'muy' son stopwords; 'están'->'estan' stopword
    assert "hojas" in tokens and "planta" in tokens and "amarillas" in tokens
    assert "las" not in tokens and "muy" not in tokens
    print("  ✓ Stopwords eliminadas, palabras útiles conservadas.")


def test_sintomas_polvo_blanco():
    print("\n─── TEST 3: 'polvo blanco' → oídio/cenicilla ───")
    sintomas = extraer_sintomas("tiene un polvo blanco en las hojas")
    print(f"  {sintomas}")
    assert "oidio" in sintomas, "Debería inferir oídio a partir de 'polvo blanco'"
    assert "cenicilla" in sintomas
    print("  ✓ Infirió oídio/cenicilla sin que el usuario usara esas palabras.")


def test_sintomas_amarillas():
    print("\n─── TEST 4: 'hojas amarillas' → amarillamiento/clorosis ───")
    sintomas = extraer_sintomas("las hojas se ven amarillas")
    print(f"  {sintomas}")
    assert "amarillamiento" in sintomas and "clorosis" in sintomas
    print("  ✓ Detectó amarillamiento/clorosis.")


def test_combinado():
    print("\n─── TEST 5: frase combinada 'hojas amarillas y polvo blanco' ───")
    sintomas = extraer_sintomas("hojas amarillas y polvo blanco")
    print(f"  {sintomas}")
    assert "amarillamiento" in sintomas
    assert "oidio" in sintomas
    # sin duplicados
    assert len(sintomas) == len(set(sintomas))
    print("  ✓ Combina varios síntomas sin duplicar.")


def test_enriquecer():
    print("\n─── TEST 6: enriquecer_consulta ───")
    consulta = enriquecer_consulta("polvo blanco en hojas")
    print(f"  '{consulta}'")
    assert "polvo blanco" in consulta
    assert "oidio" in consulta and "cenicilla" in consulta
    print("  ✓ La consulta enriquecida agrega términos canónicos.")


def test_vacio():
    print("\n─── TEST 7: texto vacío ───")
    assert extraer_sintomas("") == []
    assert extraer_sintomas("   ") == []
    print("  ✓ Texto vacío devuelve lista vacía.")


if __name__ == "__main__":
    test_limpieza()
    test_tokenizar()
    test_sintomas_polvo_blanco()
    test_sintomas_amarillas()
    test_combinado()
    test_enriquecer()
    test_vacio()
    print("\n✅ Todos los tests pasaron.\n")
