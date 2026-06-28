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

Diseño de usuario:
  - El usuario NO tiene cultivos precargados.
  - AGRICULTOR: elige qué cultivos tiene/va a cosechar (filtra la búsqueda a su parcela).
  - APRENDIZ: no elige cultivos (está aprendiendo); la búsqueda no se filtra.
  - El texto de síntomas puede ser vago ("no sé qué tiene"); el sistema funciona
    igual apoyándose en la imagen (CNN).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

from modulos import conexion

# Cultivos que el agricultor puede elegir (aquellos para los que hay documentos).
_CULTIVOS_DISPONIBLES = [
    "calabaza", "maíz", "frijol", "tomate", "fresa", "papa", "chile", "cítrico",
]


def _diagnosticar(imagen, texto, rol, cultivos_sel, forzar_offline):
    """Llama al orquestador y formatea el resultado para la interfaz."""
    from modulos.asistente import consultar

    if imagen is None:
        return "⚠ Sube una imagen de la hoja.", "", ""

    # Reglas de rol:
    #  - agricultor: usa los cultivos que eligió (lista; puede estar vacía).
    #  - aprendiz: sin cultivos (no filtra por parcela).
    cultivos = list(cultivos_sel) if rol == "agricultor" else []

    try:
        res = consultar(
            imagen=imagen,
            texto=texto or "",
            rol=rol,
            cultivos=cultivos,
            forzar_offline=True if forzar_offline else None,
        )
    except Exception as e:
        return f"❌ Error: {e}", "", ""

    diag = res["diagnostico"]

    info = []
    info.append(f"**Modo:** {res['modo'].upper()}")
    info.append(f"**CNN:** {diag['cultivo']} / {diag['enfermedad']}  "
                f"(confianza {diag['confianza_original']:.0%} → ajustada {diag['confianza_ajustada']:.0%})")
    info.append(f"**Síntomas (NLP):** {', '.join(res['sintomas']) or '— (texto vago; se usa la imagen)'}")
    info.append(f"**Fusión:** {diag['explicacion']}")
    if rol == "agricultor":
        info.append(f"**Tus cultivos:** {', '.join(cultivos) if cultivos else '(ninguno elegido)'}")
    if res["avisos"]:
        info.append("**Avisos:**")
        for a in res["avisos"]:
            info.append(f"- ⚠ {a}")
    info.append(f"**Documentos usados:** {res['n_documentos']}")
    panel = "\n\n".join(info)

    respuesta = res["respuesta"]["texto"]
    fuentes = "  \n".join(f"- {f}" for f in res["respuesta"].get("fuentes", [])) or "—"
    return panel, respuesta, fuentes


def _cambiar_rol(rol):
    """Muestra el selector de cultivos solo para el agricultor."""
    return gr.update(visible=(rol == "agricultor"), value=[])


def construir_interfaz():
    """Construye (sin lanzar) la interfaz Gradio."""
    with gr.Blocks(title="Diagnóstico Agrícola — Pruebas") as demo:
        gr.Markdown("# 🌱 Banco de pruebas — Diagnóstico Agrícola\n"
                    "Sube una foto de la hoja y, si quieres, describe los síntomas "
                    "(aunque sea vago, ej. *\"no sé qué tiene\"*).")
        gr.Markdown(f"**Conexión:** {conexion.estado_conexion()}")

        with gr.Row():
            with gr.Column(scale=1):
                rol = gr.Radio(["agricultor", "aprendiz"], value="agricultor",
                               label="¿Quién eres?",
                               info="Agricultor: eliges tus cultivos. "
                                    "Aprendiz: solo aprendes, sin cultivos.")
                cultivos_sel = gr.CheckboxGroup(
                    _CULTIVOS_DISPONIBLES,
                    label="¿Qué cultivos tienes o vas a cosechar?",
                    info="Elige los de tu parcela (filtra la búsqueda).",
                    visible=True, value=[],
                )
                img = gr.Image(type="filepath", label="Foto de la hoja")
                txt = gr.Textbox(label="Síntomas (opcional)",
                                 placeholder="ej. 'tiene polvo blanco' o 'no sé qué tiene'",
                                 lines=2)
                offline = gr.Checkbox(label="Forzar modo offline (desde caché)",
                                      value=False)
                btn = gr.Button("Diagnosticar", variant="primary")
            with gr.Column(scale=2):
                panel = gr.Markdown(label="Diagnóstico técnico")
                resp = gr.Textbox(label="Respuesta generada", lines=14)
                fuentes = gr.Markdown(label="Fuentes")

        rol.change(_cambiar_rol, inputs=rol, outputs=cultivos_sel)
        btn.click(_diagnosticar,
                  inputs=[img, txt, rol, cultivos_sel, offline],
                  outputs=[panel, resp, fuentes])

    return demo


if __name__ == "__main__":
    demo = construir_interfaz()
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
