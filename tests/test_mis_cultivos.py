"""
tests/test_mis_cultivos.py
Prueba básica del módulo mis_cultivos.
Ejecutar desde la raíz del proyecto:
    python -m tests.test_mis_cultivos
"""

import sys
from pathlib import Path

# Para poder importar desde modulos/ sin instalar el paquete
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modulos.mis_cultivos import (
    agregar,
    quitar,
    listar,
    existe,
    limpiar_todo,
)

# Usar una BD temporal para los tests
_DIR = Path(__file__).resolve().parent.parent
_BD_TEST = _DIR / "datos" / "test_mis_cultivos.db"


def limpiar():
    """Elimina el archivo de prueba si existe."""
    if _BD_TEST.exists():
        _BD_TEST.unlink()


def test_mis_cultivos():
    print("\n─── TEST 1: agregar maíz/calabaza/frijol ───")
    assert agregar("maíz", ruta_bd=_BD_TEST) is True
    assert agregar("calabaza", ruta_bd=_BD_TEST) is True
    assert agregar("frijol", ruta_bd=_BD_TEST) is True
    cultivos = listar(ruta_bd=_BD_TEST)
    print(f"  Cultivos registrados: {cultivos}")
    assert cultivos == ["calabaza", "frijol", "maíz"], (
        f"Orden/contenido inesperado: {cultivos}"
    )
    print("  ✓ Se agregaron y listaron los tres cultivos.")

    print("\n─── TEST 2: no duplica ni distingue mayúsculas ───")
    assert agregar("MAÍZ", ruta_bd=_BD_TEST) is False, "Debería detectar duplicado"
    assert agregar("  maíz  ", ruta_bd=_BD_TEST) is False, "Debería normalizar espacios"
    assert listar(ruta_bd=_BD_TEST) == ["calabaza", "frijol", "maíz"]
    print("  ✓ 'MAÍZ' y '  maíz  ' no crean duplicados.")

    print("\n─── TEST 3: existe() ───")
    assert existe("Maíz", ruta_bd=_BD_TEST) is True
    assert existe("tomate", ruta_bd=_BD_TEST) is False
    print("  ✓ existe() distingue registrados de no registrados.")

    print("\n─── TEST 4: quitar ───")
    assert quitar("calabaza", ruta_bd=_BD_TEST) is True
    assert quitar("calabaza", ruta_bd=_BD_TEST) is False, "Ya no debería estar"
    assert listar(ruta_bd=_BD_TEST) == ["frijol", "maíz"]
    print("  ✓ quitar() elimina y devuelve False si no estaba.")

    print("\n─── TEST 5: nombre vacío ───")
    assert agregar("   ", ruta_bd=_BD_TEST) is False
    assert listar(ruta_bd=_BD_TEST) == ["frijol", "maíz"]
    print("  ✓ No agrega cultivos vacíos.")

    print("\n─── TEST 6: limpiar_todo ───")
    limpiar_todo(ruta_bd=_BD_TEST)
    assert listar(ruta_bd=_BD_TEST) == []
    print("  ✓ limpiar_todo() vacía la parcela.")

    print("\n✅ Todos los tests pasaron.\n")


if __name__ == "__main__":
    limpiar()
    try:
        test_mis_cultivos()
    finally:
        limpiar()
