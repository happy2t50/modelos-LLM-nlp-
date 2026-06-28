"""
interfaz.py — Interfaz web de PRUEBAS (Gradio) para el sistema de diagnóstico.

Es un banco de pruebas independiente para testear el modelo completo (CNN + NLP +
fusión + RAG + Qwen) antes de conectarlo a la app móvil real. NO es la app final.

USO:
    pip install gradio
    set PYTHONUTF8=1
    python interfaz.py
    # abre http://localhost:7860 en el navegador

Requiere haber preparado el corpus antes (python construir_corpus.py) y, para el
modo online, Ollama corriendo con el modelo qwen3.5:0.8b.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

from modulos import mis_cultivos, conexion


def _diagnosticar(imagen, texto, rol, forzar_offline):
    """Llama al orquestador y formatea el resultado para la interfaz."""
    from modulos.asistente import consultar

    if imagen is None:
        return "⚠ Sube una imagen de la hoja.", "", ""

    try:
        res = consultar(
            imagen=imagen,
            texto=texto or "",
            rol=rol,
            forzar_offline=True if forzar_offline else None,
        )
    except Exception as e:
        return f"❌ Error: {e}", "", ""

    diag = res["diagnostico"]

    # --- Panel de diagnóstico técnico ---
    info = []
    info.append(f"**Modo:** {res['modo'].upper()}")
    info.append(f"**CNN:** {diag['cultivo']} / {diag['enfermedad']}  "
                f"(confianza {diag['confianza_original']:.0%} → ajustada {diag['confianza_ajustada']:.0%})")
    info.append(f"**Síntomas (NLP):** {', '.join(res['sintomas']) or '—'}")
    info.append(f"**Fusión:** {diag['explicacion']}")
    if res["avisos"]:
        info.append("**Avisos:**")
        for a in res["avisos"]:
            info.append(f"- ⚠ {a}")
    info.append(f"**Documentos usados:** {res['n_documentos']}")
    panel = "\n\n".join(info)

    # --- Respuesta generada ---
    respuesta = res["respuesta"]["texto"]
    fuentes = "  \n".join(f"- {f}" for f in res["respuesta"].get("fuentes", [])) or "—"

    return panel, respuesta, fuentes


def _estado_sistema():
    """Muestra el estado de cultivos registrados y la conexión."""
    try:
        cultivos = mis_cultivos.listar()
    except Exception:
        cultivos = []
    estado = conexion.estado_conexion()
    return (f"**Conexión:** {estado}  |  "
            f"**Mis cultivos:** {', '.join(cultivos) if cultivos else '(ninguno; corre construir_corpus.py)'}")


def construir_interfaz():
    """Construye (sin lanzar) la interfaz Gradio."""
    with gr.Blocks(title="Diagnóstico Agrícola — Pruebas") as demo:
        gr.Markdown("# 🌱 Banco de pruebas — Diagnóstico Agrícola\n"
                    "Sube una foto de la hoja, describe los síntomas y elige el rol.")
        gr.Markdown(_estado_sistema())

        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(type="filepath", label="Foto de la hoja")
                txt = gr.Textbox(label="Síntomas (texto del usuario)",
                                 placeholder="ej. polvo blanco como ceniza en las hojas",
                                 lines=2)
                rol = gr.Radio(["agricultor", "aprendiz"], value="agricultor",
                               label="Rol")
                offline = gr.Checkbox(label="Forzar modo offline (desde caché)",
                                      value=False)
                btn = gr.Button("Diagnosticar", variant="primary")
            with gr.Column(scale=2):
                panel = gr.Markdown(label="Diagnóstico técnico")
                resp = gr.Textbox(label="Respuesta generada", lines=14)
                fuentes = gr.Markdown(label="Fuentes")

        btn.click(_diagnosticar, inputs=[img, txt, rol, offline],
                  outputs=[panel, resp, fuentes])

    return demo


if __name__ == "__main__":
    demo = construir_interfaz()
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
