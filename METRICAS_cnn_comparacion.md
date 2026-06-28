# Comparación de 3 arquitecturas CNN (Requisito #1)

**Tarea:** clasificación de enfermedades de calabaza (5 clases).

**Dataset:** Calabaza/Original — 5 clases balanceadas (400 img/clase, 2000 en total).
Split 80/20 reproducible (semilla 42): 1600 entrenamiento / 400 validación.

**Entrenamiento:** transfer learning desde ImageNet, mismo split y mismos
hiperparámetros para los 3 (épocas=4, batch=16, lr=3e-4, img=224px, GPU CUDA).

## Resultados

| Modelo | Accuracy | F1-macro | Params (M) | Tamaño (MB) | Latencia (ms/img) |
|---|---|---|---|---|---|
| MobileNetV2 | 0.858 | **0.853** | 2.2 | 8.9 | 2.7 |
| ResNet18 | 0.828 | 0.821 | 11.2 | 44.7 | 1.9 |
| EfficientNet-B0 | 0.845 | 0.837 | 4.0 | 16.1 | 2.3 |

**Mejor F1-macro en este test:** MobileNetV2 (0.853).

## Interpretación y decisión

A esta escala reducida (1 cultivo, 5 clases, 4 épocas) los tres modelos quedan
**muy parejos** (F1 ~0.82–0.85), diferencia dentro del margen de variación para tan
pocas épocas. Observaciones:

- **MobileNetV2:** la más ligera (2.2M params, 8.9 MB) y aquí la de mejor F1 →
  excelente candidata si se quisiera correr la CNN **en el dispositivo** (modo offline).
- **EfficientNet-B0:** equilibrio sólido. Su versión escalada **EfficientNet-B4** es la
  elegida en producción porque sobre las **50 clases completas** alcanza ~97% F1, donde
  la capacidad del modelo escalado marca la diferencia (este test de 5 clases no captura
  ese escenario).
- **ResNet18:** línea base clásica; más pesada y aquí la de menor F1.

**Decisión:** EfficientNet-B4 para producción (mejor resultado en el problema completo
de 50 clases, ya entrenado en `best.pth`). MobileNetV2 queda como **alternativa ligera
recomendada** si se necesita inferencia de imagen on-device. A esta escala la diferencia
es pequeña, por lo que la elección se sustenta en el comportamiento sobre el problema
completo, no solo en este subconjunto.

> **Honestidad metodológica:** esta es una comparación a escala reducida para justificar
> de forma reproducible la elección de arquitectura. No reentrena el modelo de 50 clases;
> ese ya existe (`best.pth`, EfficientNet-B4, ~97% F1).

## Reproducir

```powershell
set PYTHONUTF8=1
python comparar_cnns.py
```
