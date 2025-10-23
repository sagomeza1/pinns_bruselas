# Estructura del Proyecto

```bash
nombre_proyecto/
│
├── data/
│   ├── raw/                # Datos originales, sin procesar
│   ├── interim/            # Datos intermedios o transformados parcialmente
│   └── processed/          # Datos finales listos para el modelado
│
├── notebooks/
│   ├── 01_exploracion.ipynb       # Análisis exploratorio (EDA)
│   ├── 02_preprocesamiento.ipynb  # Limpieza, normalización, encoding
│   ├── 03_entrenamiento.ipynb     # Pruebas de modelos iniciales
│   └── 04_evaluacion.ipynb        # Métricas, gráficos, comparaciones
│
├── src/                   # Código fuente (Python)
│   ├── __init__.py
│   ├── data/
│   │   ├── make_dataset.py       # Lectura y procesamiento de datos
│   │   ├── augmentations.py      # Aumentos de datos si aplica (imágenes, etc.)
│   │   └── loaders.py            # Clases DataLoader / Dataset
│   ├── models/
│   │   ├── architecture.py       # Definición de la red neuronal (PyTorch, TF, etc.)
│   │   ├── train.py              # Funciones de entrenamiento y validación
│   │   └── evaluate.py           # Métricas, curvas ROC, etc.
│   ├── utils/
│   │   ├── config.py             # Parámetros globales, rutas, seeds, etc.
│   │   ├── visualization.py      # Gráficos y reportes
│   │   └── metrics.py            # Funciones de evaluación personalizadas
│   └── main.py                   # Script principal de entrenamiento
│
├── models/
│   ├── checkpoints/        # Pesos guardados durante el entrenamiento
│   ├── final/              # Modelos finales exportados (.pt, .h5, .onnx)
│   └── logs/               # TensorBoard o archivos de seguimiento de entrenamiento
│
├── tests/
│   ├── test_data.py        # Pruebas unitarias sobre procesamiento de datos
│   ├── test_model.py       # Pruebas sobre arquitectura y entrenamiento
│   └── test_utils.py
│
├── reports/
│   ├── figures/            # Gráficos y visualizaciones
│   └── informe_final.md    # Resultados, conclusiones o documentación técnica
│
├── requirements.txt        # Dependencias (o environment.yml si usas conda)
├── config.yaml             # Hiperparámetros y rutas centralizadas
├── .gitignore
├── README.md               # Descripción del proyecto
└── LICENSE
