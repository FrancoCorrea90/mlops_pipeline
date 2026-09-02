# mlops_pipeline

Proyecto integrador — Módulo 5 (Henry). Pipeline de datos y modelado para predecir el pago a tiempo de créditos otorgados por una empresa financiera, a partir de datos históricos de crédito.

## Estructura

```
mlops_pipeline/
├── Base_de_datos.xlsx        # Fuente de datos original (histórico de créditos)
├── requirements.txt          # Dependencias del entorno virtual
└── src/
    ├── config.json            # Configuración del proyecto (nombre del entorno virtual)
    ├── Cargar_datos.ipynb      # Carga y validación inicial de Base_de_datos.xlsx
    └── comprension_eda.ipynb   # Análisis exploratorio de datos (EDA)
```

## Cómo ejecutar

1. Crear y activar un entorno virtual con las dependencias de `requirements.txt`, y registrarlo como kernel de Jupyter.
2. Abrir `src/Cargar_datos.ipynb` y ejecutar todas las celdas para validar la carga de `Base_de_datos.xlsx`.
3. Abrir `src/comprension_eda.ipynb` y ejecutar todas las celdas para el análisis exploratorio (nulos, outliers, distribuciones, correlaciones y relación con la variable objetivo `Pago_atiempo`).

## Estado

Avance 1: carga de datos y análisis exploratorio. Próximos pasos: limpieza de datos y desarrollo del modelo predictivo.
