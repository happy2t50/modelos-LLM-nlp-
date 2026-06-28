"""
ejecutar.py — Punto de entrada para probar el sistema completo de diagnóstico.

Une todas las fases: CNN (imagen) + NLP (texto) + fusión + recuperación
(TF-IDF/BERT) + caché Top-K + generación con Qwen, según el rol del usuario.

USO RÁPIDO
----------
1) Preparar el almacén una sola vez (carga documentos/, construye índices y
   registra tus cultivos):

       python ejecutar.py --preparar

2) Hacer una consulta con una foto y un texto de síntomas:

       python ejecutar.py --imagen "ruta/a/hoja.jpg" --texto "polvo blanco en las hojas" --rol agricultor

   Roles: agricultor (respuesta simple) | aprendiz (respuesta técnica).

3) Forzar modo offline (probar sin internet, desde el caché):

       python ejecutar.py --imagen "..." --texto "..." --offline

NOTAS
-----
- En modo online genera con Qwen vía Ollama. Asegúrate de tener Ollama corriendo
  (ollama serve) y el modelo descargado (ollama pull qwen3.5:0.8b).
- En Windows, si ves errores de acentos en la consola, ejecuta antes:
      set PYTHONUTF8=1
"""

import sys
import argparse
from pathlib import Path

# Permite ejecutar como script suelto (sin instalar el paquete)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from modulos import mis_cultivos
from modulos.almacen_documentos import cargar_desde_directorio, construir_indice, listar_documentos
from modulos.busqueda_semantica import construir_embeddings

# Cultivos por defecto de la parcela (puedes cambiarlos con --cultivos)
_CULTIVOS_DEFECTO = ["maíz", "calabaza", "frijol"]


def preparar(cultivos: list[str]) -> None:
    """Carga documentos, construye índices/embeddings y registra los cultivos."""
    print("== Preparando el almacén ==")
    n = cargar_desde_directorio()
    print(f"  Documentos nuevos cargados: {n}")
    docs = listar_documentos()
    print(f"  Documentos en la base: {len(docs)}")
    for d in docs:
        print(f"    - {d['cultivo']} / {d['enfermedad']}")

    print("  Construyendo índice TF-IDF...")
    construir_indice()
    print("  Construyendo embeddings BERT...")
    construir_embeddings()

    mis_cultivos.limpiar_todo()
    for c in cultivos:
        mis_cultivos.agregar(c)
    print(f"  Cultivos registrados (mis_cultivos): {mis_cultivos.listar()}")
    print("== Almacén listo ==\n")


def consultar_y_mostrar(imagen: str, texto: str, rol: str, offline: bool) -> None:
    """Ejecuta una consulta completa y muestra el resultado de forma legible."""
    # Import tardío: cargar la CNN/orquestador solo cuando se va a consultar.
    from modulos.asistente import consultar

    forzar_offline = True if offline else None  # None = detectar internet

    print("== Consulta ==")
    print(f"  Imagen: {imagen}")
    print(f"  Texto:  {texto}")
    print(f"  Rol:    {rol}\n")

    res = consultar(
        imagen=imagen,
        texto=texto,
        rol=rol,
        forzar_offline=forzar_offline,
    )

    diag = res["diagnostico"]
    print("─" * 60)
    print(f"MODO: {res['modo'].upper()}")
    print(f"CNN  : {diag['cultivo']} / {diag['enfermedad']}  "
          f"(confianza {diag['confianza_original']:.0%} → ajustada {diag['confianza_ajustada']:.0%})")
    print(f"NLP  : {res['sintomas']}")
    print(f"Fusión: {diag['explicacion']}")
    if res["avisos"]:
        print("AVISOS:")
        for a in res["avisos"]:
            print(f"  ⚠ {a}")
    print(f"Documentos usados: {res['n_documentos']}  "
          f"{[d['cultivo'] + '/' + d['enfermedad'] for d in res['documentos']]}")
    print("─" * 60)
    print("RESPUESTA:\n")
    print(res["respuesta"]["texto"])
    print()
    if res["respuesta"].get("fuentes"):
        print("Fuentes:", "; ".join(res["respuesta"]["fuentes"]))


def main():
    p = argparse.ArgumentParser(
        description="Sistema de diagnóstico agrícola (CNN + NLP + RAG + Qwen).")
    p.add_argument("--preparar", action="store_true",
                   help="Construye el almacén (documentos, índices, embeddings, cultivos).")
    p.add_argument("--imagen", help="Ruta de la foto de la hoja.")
    p.add_argument("--texto", default="", help="Texto de síntomas del usuario.")
    p.add_argument("--rol", default="agricultor", choices=["agricultor", "aprendiz"],
                   help="Nivel de la respuesta.")
    p.add_argument("--offline", action="store_true",
                   help="Forzar modo offline (responder desde el caché).")
    p.add_argument("--cultivos", nargs="*", default=_CULTIVOS_DEFECTO,
                   help="Cultivos de la parcela (con --preparar).")
    args = p.parse_args()

    if args.preparar:
        preparar(args.cultivos)

    if args.imagen:
        consultar_y_mostrar(args.imagen, args.texto, args.rol, args.offline)
    elif not args.preparar:
        p.print_help()
        print("\nSugerencia: primero 'python ejecutar.py --preparar', "
              "luego una consulta con --imagen y --texto.")


if __name__ == "__main__":
    main()
