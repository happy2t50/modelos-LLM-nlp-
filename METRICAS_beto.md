# Fine-tuning de BETO (Requisito #3)

**Modelo base:** dccuchile/bert-base-spanish-wwm-uncased (BETO, BERT en español)

**Tarea:** clasificación de texto agrícola → cultivo (9 clases).

**Datos:** 1815 fragmentos del corpus combinado (1452 entrenamiento / 363 prueba, split estratificado 80/20).

**Hiperparámetros:** épocas=4, batch=16, lr=2e-05, max_len=160, dispositivo=cuda.

## Resultados

- **Accuracy:** 0.868
- **F1-macro:** 0.868

### Reporte por clase

```
              precision    recall  f1-score   support

    calabaza      0.902     0.822     0.860        90
       chile      0.867     0.765     0.812        17
     cítrico      0.882     0.938     0.909        32
       fresa      0.660     0.786     0.717        42
      frijol      0.870     0.952     0.909        21
     general      0.938     0.984     0.961        62
        maíz      0.877     0.851     0.864        67
        papa      0.846     1.000     0.917        11
      tomate      1.000     0.762     0.865        21

    accuracy                          0.868       363
   macro avg      0.871     0.873     0.868       363
weighted avg      0.874     0.868     0.868       363

```

## Reproducir

```powershell
set PYTHONUTF8=1
python finetune_beto.py
```
