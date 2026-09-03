💳 Predicción de Comportamiento de Pago - MLOps Pipeline
Proyecto integrador desarrollado en el marco del Módulo 5 - Producción ML / MLOps.

El proyecto tiene como objetivo construir progresivamente un pipeline de Machine Learning capaz de predecir el comportamiento de pago de clientes de crédito, incorporando buenas prácticas de análisis de datos, ingeniería de características, entrenamiento, evaluación, versionado y posterior preparación para producción.

📋 Caso de negocio
En el otorgamiento de créditos resulta importante poder anticipar el comportamiento de pago de los clientes.

A partir de información histórica relacionada con características crediticias, laborales y financieras, se busca desarrollar un modelo de clasificación que permita estimar si un cliente realizará sus pagos a tiempo.

La variable objetivo del proyecto es:

Pago_atiempo
donde:

1 = Pago a tiempo
0 = No pago a tiempo
El objetivo final no es solamente obtener un modelo predictivo, sino construir un flujo reproducible y organizado que permita evolucionar progresivamente hacia un entorno MLOps.

🗂️ Estructura del proyecto
mlops_pipeline/
│
├── src/
│   ├── comprension_eda.ipynb
│   ├── ft_engineering.py
│   └── model_train_evaluation.py
│
├── Base_de_datos.xlsx
├── requirements.txt
├── .gitignore
└── README.md
La estructura será ampliada a medida que se incorporen nuevas etapas del proyecto.

🔍 1. Comprensión y análisis de datos
El análisis exploratorio inicial permitió conocer la estructura y principales características del dataset.

El conjunto de datos contiene:

10.763 registros
23 variables
No se detectaron filas duplicadas.

Durante esta etapa se analizaron:

tipos de variables;
valores faltantes;
distribución de las características;
comportamiento de la variable objetivo;
calidad general de los datos;
posibles transformaciones necesarias para el modelado.
Variable objetivo
La distribución de Pago_atiempo presenta un desbalance importante:

Pago a tiempo       ≈ 95 %
No pago a tiempo    ≈ 5 %
Esta característica será especialmente considerada durante la evaluación de los modelos, evitando utilizar únicamente accuracy como criterio de selección.

🛠️ 2. Ingeniería de características - v1.1.0
La preparación de datos se implementa en:

src/ft_engineering.py
El componente realiza las transformaciones necesarias para preparar los datos antes del entrenamiento.

Entre las principales tareas se encuentran:

limpieza y validación de variables;
tratamiento de valores faltantes;
transformación de variables categóricas;
tratamiento de variables ordinales;
generación de nuevas características a partir de fechas;
separación entre variables predictoras y variable objetivo;
división estratificada entre entrenamiento y prueba;
construcción de un pipeline de preprocesamiento mediante ColumnTransformer.
El pipeline utiliza herramientas de Scikit-learn como:

SimpleImputer
OneHotEncoder
OrdinalEncoder
ColumnTransformer
Pipeline
El preprocesamiento se ajusta exclusivamente sobre los datos de entrenamiento para evitar fuga de información entre train y test.

⚠️ Consideraciones sobre los datos
El dataset fue proporcionado sin un diccionario de datos completo.

Por este motivo, algunas variables requieren trabajar bajo supuestos documentados.

En particular, se identificaron variables asociadas a saldos cuyo momento de generación no puede determinarse con certeza:

saldo_mora
saldo_total
saldo_principal
saldo_mora_codeudor
Ante la posibilidad de que contengan información posterior al otorgamiento del crédito, fueron consideradas como variables con potencial data leakage y excluidas preventivamente del modelo principal.

Posteriormente podrá realizarse un análisis comparativo para evaluar su impacto sobre el desempeño del modelo.

También se detectaron inconsistencias en tendencia_ingresos, donde coexistían categorías válidas con valores numéricos. Estos registros fueron tratados dentro del proceso de limpieza antes de la codificación de la variable.

✂️ División de los datos
Se utiliza una división:

80 % entrenamiento
20 % prueba
con estratificación sobre la variable objetivo.

Esto permite conservar aproximadamente la misma proporción de clases en ambos conjuntos.

Resultado:

Train: 8.610 registros
Test:  2.153 registros
Luego del preprocesamiento se obtienen conjuntos sin valores faltantes y con una estructura consistente para el entrenamiento.

🤖 3. Entrenamiento y evaluación - v1.0.1
🚧 En desarrollo

La siguiente etapa del proyecto estará orientada al entrenamiento y comparación de diferentes algoritmos de clasificación supervisada.

Se implementará el componente:

src/model_train_evaluation.py
La evaluación tendrá especialmente en cuenta el fuerte desbalance existente en la variable objetivo.

Por este motivo, además de accuracy, se analizarán métricas como:

Precision
Recall
F1-Score
ROC-AUC
Matriz de confusión
El objetivo será seleccionar un modelo candidato teniendo en cuenta tanto el desempeño estadístico como el objetivo de negocio.

📊 4. Comparación y selección del modelo
⏳ Pendiente

Los modelos entrenados serán comparados mediante tablas y visualizaciones.

La selección final no estará basada exclusivamente en la métrica con mayor valor general, sino en la capacidad del modelo para identificar correctamente los casos relevantes para el problema de riesgo crediticio.

Esta sección será actualizada con los resultados obtenidos durante el modelamiento.

🔄 5. Evolución hacia MLOps
⏳ Próximas etapas

El proyecto continuará incorporando progresivamente componentes asociados al ciclo de vida de Machine Learning:

Datos
  ↓
EDA
  ↓
Feature Engineering
  ↓
Entrenamiento
  ↓
Evaluación
  ↓
Selección del modelo
  ↓
Versionado
  ↓
Producción
  ↓
Monitoreo
El objetivo es que cada etapa pueda integrarse de forma modular, reproducible y trazable.

🌿 Flujo de trabajo con Git
El repositorio utiliza diferentes ramas para separar desarrollo, integración y versiones estables.

feature/*
    ↓
developer
    ↓
main
feature/*
Ramas temporales utilizadas para desarrollar funcionalidades específicas.

developer
Rama de integración de los diferentes componentes del proyecto.

main
Contiene las versiones estables y aprobadas.

Los cambios se incorporan mediante Pull Requests, permitiendo mantener trazabilidad sobre la evolución del proyecto.

⚙️ Instalación
Clonar el repositorio:

git clone https://github.com/FrancoCorrea90/mlops_pipeline.git
Ingresar al proyecto:

cd mlops_pipeline
Crear un entorno virtual:

python -m venv .venv
Activarlo en Windows:

.\.venv\Scripts\Activate.ps1
Instalar las dependencias:

pip install -r requirements.txt
▶️ Ejecución
Actualmente puede ejecutarse el pipeline de Feature Engineering mediante:

python src/ft_engineering.py
Los siguientes componentes se irán incorporando a medida que avance el desarrollo.

🧰 Tecnologías utilizadas
Python
Pandas
NumPy
Scikit-learn
Jupyter Notebook
Git
GitHub
🎓 Contexto
Proyecto Integrador
Módulo 5 - Producción ML / MLOps

El proyecto se desarrolla de manera incremental, aplicando conceptos de Machine Learning, ingeniería de características, reproducibilidad, control de versiones y buenas prácticas de MLOps.
