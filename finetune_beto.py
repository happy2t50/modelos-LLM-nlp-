"""
finetune_beto.py — Fine-tuning de BETO (Requisito #3).

Tarea (clasificación de texto, una de las tareas vistas en clase): dado un texto
agrícola, predecir el CULTIVO al que pertenece (9 clases). Se fine-tunea BETO
(BERT en español, dccuchile/bert-base-spanish-wwm-uncased) sobre los fragmentos
etiquetados del corpus combinado.

Por qué esta tarea: es la que tiene datos etiquetados disponibles (cada fragmento
del corpus trae su cultivo). Demuestra el fine-tuning de un transformer; la misma
receta sirve para texto-de-síntomas → enfermedad cuando haya ese dataset etiquetado.

USO (requiere GPU recomendada; corre en CPU pero lento):
    pip install transformers
    set PYTHONUTF8=1
    python finetune_beto.py

Salida:
    modelo_beto/            → modelo y tokenizer fine-tuneados
    METRICAS_beto.md        → reporte de métricas (accuracy, F1 por clase)
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

_DIR = Path(__file__).resolve().parent
_RUTA_CORPUS = _DIR / "datos" / "corpus_combinado.json"
_DIR_MODELO = _DIR / "modelo_beto"
_RUTA_REPORTE = _DIR / "METRICAS_beto.md"

_MODELO_BASE = "dccuchile/bert-base-spanish-wwm-uncased"  # BETO
_MAX_LEN = 160
_BATCH = 16
_EPOCAS = 4
_LR = 2e-5
_SEMILLA = 42

_dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fijar_semilla(s=_SEMILLA):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


class CorpusDataset(Dataset):
    """Dataset de textos tokenizados con su etiqueta de cultivo."""
    def __init__(self, textos, etiquetas, tokenizer):
        self.enc = tokenizer(textos, truncation=True, padding="max_length",
                             max_length=_MAX_LEN, return_tensors="pt")
        self.y = torch.tensor(etiquetas, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.y[i]
        return item


def cargar_datos():
    """Lee el corpus combinado → (textos, etiquetas-cultivo)."""
    data = json.load(open(_RUTA_CORPUS, encoding="utf-8"))
    textos, cultivos = [], []
    for r in data:
        t = (r.get("contenido_texto") or "").strip()
        c = (r.get("cultivo") or "").strip()
        if len(t) > 40 and c:
            textos.append(t)
            cultivos.append(c)
    return textos, cultivos


def main():
    _fijar_semilla()
    print(f"Dispositivo: {_dispositivo}")

    textos, cultivos = cargar_datos()
    clases = sorted(set(cultivos))
    cls2id = {c: i for i, c in enumerate(clases)}
    id2cls = {i: c for c, i in cls2id.items()}
    y = [cls2id[c] for c in cultivos]
    print(f"Ejemplos: {len(textos)} | Clases ({len(clases)}): {clases}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        textos, y, test_size=0.2, random_state=_SEMILLA, stratify=y)
    print(f"Entrenamiento: {len(X_tr)} | Prueba: {len(X_te)}")

    print(f"\nCargando BETO ({_MODELO_BASE})...")
    tokenizer = AutoTokenizer.from_pretrained(_MODELO_BASE)
    modelo = AutoModelForSequenceClassification.from_pretrained(
        _MODELO_BASE, num_labels=len(clases), use_safetensors=True).to(_dispositivo)

    ds_tr = CorpusDataset(X_tr, y_tr, tokenizer)
    ds_te = CorpusDataset(X_te, y_te, tokenizer)
    dl_tr = DataLoader(ds_tr, batch_size=_BATCH, shuffle=True)
    dl_te = DataLoader(ds_te, batch_size=_BATCH)

    optim = torch.optim.AdamW(modelo.parameters(), lr=_LR)

    print("\n== Entrenando ==")
    for epoca in range(1, _EPOCAS + 1):
        modelo.train()
        perdida_total = 0.0
        for batch in dl_tr:
            batch = {k: v.to(_dispositivo) for k, v in batch.items()}
            optim.zero_grad()
            out = modelo(**batch)
            out.loss.backward()
            optim.step()
            perdida_total += out.loss.item()
        print(f"  Época {epoca}/{_EPOCAS} — pérdida media: {perdida_total/len(dl_tr):.4f}")

    print("\n== Evaluando ==")
    modelo.eval()
    pred, real = [], []
    with torch.no_grad():
        for batch in dl_te:
            labels = batch.pop("labels")
            batch = {k: v.to(_dispositivo) for k, v in batch.items()}
            logits = modelo(**batch).logits
            pred.extend(logits.argmax(dim=1).cpu().numpy())
            real.extend(labels.numpy())

    acc = accuracy_score(real, pred)
    f1m = f1_score(real, pred, average="macro")
    reporte = classification_report(real, pred, target_names=clases, digits=3, zero_division=0)
    print(f"\nAccuracy: {acc:.3f} | F1-macro: {f1m:.3f}\n")
    print(reporte)

    # Guardar modelo + tokenizer + mapa de clases
    _DIR_MODELO.mkdir(exist_ok=True)
    modelo.save_pretrained(_DIR_MODELO)
    tokenizer.save_pretrained(_DIR_MODELO)
    json.dump({"cls2id": cls2id, "id2cls": id2cls},
              open(_DIR_MODELO / "clases.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\nModelo guardado en {_DIR_MODELO.name}/")

    # Reporte markdown
    with open(_RUTA_REPORTE, "w", encoding="utf-8") as f:
        f.write("# Fine-tuning de BETO (Requisito #3)\n\n")
        f.write(f"**Modelo base:** {_MODELO_BASE} (BETO, BERT en español)\n\n")
        f.write("**Tarea:** clasificación de texto agrícola → cultivo "
                f"({len(clases)} clases).\n\n")
        f.write(f"**Datos:** {len(textos)} fragmentos del corpus combinado "
                f"({len(X_tr)} entrenamiento / {len(X_te)} prueba, split estratificado 80/20).\n\n")
        f.write(f"**Hiperparámetros:** épocas={_EPOCAS}, batch={_BATCH}, lr={_LR}, "
                f"max_len={_MAX_LEN}, dispositivo={_dispositivo.type}.\n\n")
        f.write(f"## Resultados\n\n")
        f.write(f"- **Accuracy:** {acc:.3f}\n")
        f.write(f"- **F1-macro:** {f1m:.3f}\n\n")
        f.write("### Reporte por clase\n\n```\n" + reporte + "\n```\n\n")
        f.write("## Reproducir\n\n```powershell\nset PYTHONUTF8=1\npython finetune_beto.py\n```\n")
    print(f"Reporte guardado en {_RUTA_REPORTE.name}")


if __name__ == "__main__":
    main()
