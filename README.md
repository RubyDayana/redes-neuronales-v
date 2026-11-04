# Redes Neuronales V — Reconocimiento facial con redes siamesas y Triplet Loss

CADI: Deep Learning — Conceptos · Universidad de Cundinamarca
Estudiante: Ruby Dayana Cárdenas Gómez · Docente: Nidia Stella García Roa

Red siamesa para reconocimiento facial (verificación, identificación, one-shot, few-shot y zero-shot) entrenada con **Triplet Loss** en **PyTorch** y en **TensorFlow + Keras**, con un dataset propio de 38 rostros de 3 personas que autorizaron el uso de sus fotos para fines académicos.

| Notebook | Colab |
|---|---|
| 01 Dataset y preprocesamiento (MTCNN, alineación, recorte) | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RubyDayana/redes-neuronales-v/blob/main/01%20DatasetFacialPreprocesamiento.ipynb) |
| 02 Siamesa en PyTorch + Triplet Loss | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RubyDayana/redes-neuronales-v/blob/main/02%20SiamesaPyTorchTripletLoss.ipynb) |
| 03 Siamesa en Keras + Triplet Loss | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RubyDayana/redes-neuronales-v/blob/main/03%20SiamesaKerasTripletLoss.ipynb) |

## Resultados (10 repartos, fotos de prueba, promedio ± desv.)

| Modelo | AUC verificación | Identificación (bal.) | 1-shot |
|---|---|---|---|
| Píxeles crudos | 0.615 ± 0.119 | 0.629 ± 0.161 | 0.489 |
| Red sin entrenar | 0.638 ± 0.089 | 0.569 ± 0.197 | 0.502 |
| Siamesa PyTorch | 0.772 ± 0.082 | 0.682 ± 0.123 | 0.704 |
| Siamesa Keras | 0.765 ± 0.118 | 0.758 ± 0.115 | 0.759 |

Zero-shot: **no se logró** (35.5 % frente a 33.3 % de azar) porque la red se entrena con solo dos identidades. Batch-hard colapsó; se usa batch-all.

## Estructura
- `datos/rostros/` rostros alineados 112×112 (una carpeta por persona) y `datos/metadatos.csv`
- `modelos/` pesos entrenados (.pt, .keras) · `resultados/` métricas por reparto (.csv)

## Cómo ejecutar
Abrir los notebooks en Colab en orden 01 → 02 → 03 (el 01 necesita las fotos originales en `datos/fotos/<persona>/`, que no se publican; los notebooks 02 y 03 solo necesitan `datos/rostros/`). MTCNN: `pip install facenet-pytorch --no-deps`.

## Privacidad
Las fotos originales no se incluyen (traen metadatos de ubicación). Los rostros se usan solo para este trabajo académico, con autorización de las personas.
