\# 💳 Predicción de Comportamiento de Pago - MLOps Pipeline



Proyecto integrador desarrollado en el marco del \*\*Módulo 5 - Producción ML / MLOps\*\* del bootcamp de Henry.



El objetivo del proyecto es construir progresivamente un pipeline de Machine Learning capaz de predecir el comportamiento de pago de clientes de crédito, incorporando buenas prácticas de análisis de datos, ingeniería de características, entrenamiento, evaluación, versionado, monitoreo y preparación para producción.



\---



\## 📋 Caso de negocio



En el otorgamiento de créditos resulta importante poder anticipar el comportamiento de pago de los clientes.



A partir de información histórica relacionada con características crediticias, laborales y financieras, se desarrolla un modelo de clasificación que permite estimar si un cliente realizará sus pagos a tiempo.



La variable objetivo es:



```text

Pago\_atiempo

```



donde:



\- `1` = Pago a tiempo.

\- `0` = No paga a tiempo.



El objetivo no es únicamente obtener un modelo predictivo, sino construir un flujo reproducible y organizado que pueda evolucionar progresivamente hacia un entorno MLOps.



\---



\## 🗂️ Estructura actual del proyecto



```text

mlops\_pipeline/

│

├── src/

│   ├── Cargar\_datos.ipynb

│   ├── comprension\_eda.ipynb

│   ├── ft\_engineering.py

│   ├── model\_train\_evaluation.py

│   └── model\_monitoring.py

│

├── Base\_de\_datos.xlsx

├── requirements.txt

├── .gitignore

└── README.md

```



El archivo `best\_model.joblib` se genera localmente al ejecutar el entrenamiento, pero se encuentra excluido del repositorio mediante `.gitignore` porque es un artefacto generado y no forma parte de la estructura fuente del entregable.



\---



\## 🔎 1. Comprensión y análisis exploratorio de datos



El análisis exploratorio inicial permitió conocer la estructura y las principales características del dataset.



El conjunto de datos contiene:



\- \*\*10.763 registros\*\*.

\- \*\*23 variables\*\*.

\- No se detectaron filas duplicadas.



Durante esta etapa se analizaron tipos de variables, valores faltantes, distribución de las características, comportamiento de la variable objetivo, calidad general de los datos y posibles transformaciones necesarias para el modelado.



\### Variable objetivo



La distribución de `Pago\_atiempo` presenta un fuerte desbalance:



```text

Pago a tiempo       ≈ 95 %

No pago a tiempo    ≈ 5 %

```



Por este motivo, la evaluación de los modelos no se basa únicamente en `Accuracy`. Se presta especial atención al desempeño sobre la clase `0`, que representa a los clientes que no pagan a tiempo.



\---



\## 🛠️ 2. Ingeniería de características



La preparación de datos se implementa en:



```text

src/ft\_engineering.py

```



Entre las principales tareas se encuentran:



\- limpieza y validación de variables;

\- tratamiento de valores faltantes;

\- transformación de variables categóricas;

\- tratamiento de variables ordinales;

\- generación de nuevas características a partir de la fecha del préstamo;

\- separación entre variables predictoras y variable objetivo;

\- división estratificada entre entrenamiento y prueba;

\- construcción de un pipeline de preprocesamiento mediante `ColumnTransformer`.



El pipeline utiliza `SimpleImputer`, `OneHotEncoder`, `OrdinalEncoder`, `ColumnTransformer` y `Pipeline` de Scikit-learn.



El preprocesamiento se ajusta exclusivamente sobre los datos de entrenamiento para evitar fuga de información entre `train` y `test`.



\### Consideraciones sobre los datos y posible leakage



El dataset fue proporcionado sin un diccionario de datos completo. Por ese motivo, algunas variables requieren trabajar bajo supuestos documentados.



Se identificaron variables cuyo momento de generación no puede determinarse con certeza:



```text

puntaje

saldo\_mora

saldo\_total

saldo\_principal

saldo\_mora\_codeudor

```



Ante la posibilidad de que contengan información posterior al otorgamiento del crédito, fueron consideradas variables con potencial `data leakage` y excluidas preventivamente del modelo principal.



También se detectaron inconsistencias en `tendencia\_ingresos`, donde coexistían categorías válidas con valores no esperados. Estos registros son tratados dentro del proceso de limpieza previo a la codificación.



\### División de los datos



Se utiliza una división estratificada:



```text

80 % entrenamiento

20 % prueba

```



Resultado:



```text

Train: 8.610 registros

Test:  2.153 registros

```



La estratificación permite mantener aproximadamente la misma proporción de clases en ambos conjuntos.



\---



\## 🤖 3. Entrenamiento y evaluación de modelos



El entrenamiento se implementa en:



```text

src/model\_train\_evaluation.py

```



Se compararon tres algoritmos de clasificación supervisada:



\- Logistic Regression;

\- Random Forest;

\- XGBoost.



Debido al fuerte desbalance de la variable objetivo, se analizaron `Accuracy`, `Balanced Accuracy`, Precision de la clase `0`, Recall de la clase `0`, F1-Score de la clase `0`, ROC-AUC de la clase `0`, PR-AUC de la clase `0` y matrices de confusión.



\### Resultados comparativos



| Modelo | Accuracy | Balanced Accuracy | Precision clase 0 | Recall clase 0 | F1 clase 0 | ROC-AUC clase 0 | PR-AUC clase 0 |

|---|---:|---:|---:|---:|---:|---:|---:|

| Logistic Regression | 0.6521 | 0.5752 | 0.0669 | 0.4902 | 0.1178 | 0.6200 | 0.0786 |

| Random Forest | 0.9526 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.6700 | 0.1441 |

| XGBoost | 0.8402 | 0.5994 | 0.1097 | 0.3333 | \*\*0.1650\*\* | \*\*0.6787\*\* | \*\*0.1485\*\* |



\### Modelo seleccionado



Se seleccionó \*\*XGBoost\*\* como modelo candidato porque obtuvo el mejor `F1-Score` sobre la clase minoritaria (`0`).



La selección no se basa exclusivamente en `Accuracy`, ya que Random Forest alcanza una Accuracy elevada pero no identifica correctamente los casos de no pago a tiempo. Logistic Regression logra un Recall superior para la clase `0`, pero genera una cantidad considerablemente mayor de falsos positivos. XGBoost presenta un mejor equilibrio entre Precision y Recall sobre la clase relevante para el problema.



Al finalizar el entrenamiento, el modelo seleccionado se persiste localmente como:



```text

best\_model.joblib

```



Este archivo es utilizado por la aplicación de monitoreo y se regenera ejecutando nuevamente el entrenamiento.



\---



\## 📊 4. Monitoreo de Data Drift - Avance 3



El monitoreo se implementa en:



```text

src/model\_monitoring.py

```



El objetivo es detectar cambios en la distribución de las variables de entrada que puedan indicar que la población actual se está alejando de la población utilizada como referencia.



\### Métricas utilizadas



Para variables numéricas se calculan:



\- \*\*Kolmogorov-Smirnov (KS)\*\*;

\- \*\*Population Stability Index (PSI)\*\*;

\- \*\*Jensen-Shannon Divergence\*\*.



Para variables categóricas se calculan:



\- \*\*Chi-cuadrado\*\*;

\- \*\*Population Stability Index (PSI)\*\*;

\- \*\*Jensen-Shannon Divergence\*\*.



Las pruebas estadísticas se utilizan como evidencia complementaria. El semáforo operativo principal se construye a partir de PSI.



\### Umbrales de PSI



Para este ejercicio se adoptan los siguientes umbrales operativos:



```text

PSI < 0.10          → Estable

0.10 ≤ PSI < 0.25   → Advertencia

PSI ≥ 0.25          → Crítico

```



\---



\## 🧪 Metodología de referencia y población actual



En un escenario productivo ideal, el monitoreo compara una población histórica de referencia con datos nuevos posteriores al despliegue.



En este proyecto no se dispone de un segundo dataset independiente correspondiente a observaciones posteriores a producción. Por ese motivo, el modo de monitoreo por defecto utiliza:



```text

TRAIN → población histórica de referencia

TEST  → población actual simulada

```



Esta estrategia permite validar el funcionamiento del sistema de detección de Data Drift utilizando observaciones que no fueron empleadas para ajustar el modelo.



\*\*Importante:\*\* los resultados obtenidos en este modo deben interpretarse como una \*\*simulación académica del proceso de monitoreo\*\* y no como evidencia de drift real posterior al despliegue.



La implementación permite además cargar desde Streamlit un archivo externo en formato CSV o Excel y utilizarlo como nueva población actual. De esta manera, cuando exista un lote real de datos posteriores al despliegue, puede reemplazarse la muestra `TEST` sin modificar las funciones de cálculo de drift.



\---



\## 📅 Monitoreo temporal



Se definió una periodicidad de análisis \*\*mensual\*\*.



Para evitar conclusiones basadas en muestras demasiado pequeñas, solo se informan períodos que contienen al menos:



```text

50 observaciones

```



Las variables `anio\_prestamo` y `mes\_prestamo` se utilizan para identificar cada período mensual, pero se excluyen del cálculo de drift dentro de ese mismo período. Esto evita detectar artificialmente cambios provocados por la propia segmentación temporal.



El monitoreo temporal informa cantidad de observaciones por período, PSI promedio, PSI máximo, variable más afectada, estado del período, variables en advertencia, variables críticas, cantidad total de variables con alerta y recomendación operativa.



\---



\## 📈 Resultados del monitoreo



\### Análisis global



Al comparar `TRAIN` contra `TEST`, las variables presentan valores de PSI global inferiores a `0.10`.



El mayor PSI global observado fue aproximadamente:



```text

0.0164

```



correspondiente a `promedio\_ingresos\_datacredito`.



Por lo tanto, bajo la simulación global `TRAIN vs TEST`, no se detectaron desviaciones relevantes según el criterio de PSI.



En `huella\_consulta` se obtuvo un `KS p-value` inferior a `0.05`, aunque su PSI y su divergencia Jensen-Shannon fueron muy bajos. Por ese motivo, la diferencia estadística no se interpreta por sí sola como un cambio de magnitud práctica relevante.



\### Análisis temporal



Al analizar la población `TEST` por períodos mensuales sí aparecen cambios locales respecto de la referencia histórica general.



La variable `promedio\_ingresos\_datacredito` resultó la más afectada en varios períodos. Los mayores valores de PSI temporal se observaron, entre otros, en junio y septiembre de 2025.



Estos resultados no implican necesariamente que el modelo haya perdido capacidad predictiva. El Data Drift describe cambios en las variables de entrada; una degradación real del modelo requeriría además evaluar el rendimiento predictivo sobre datos posteriores al despliegue cuando se disponga de sus valores reales.



\---



\## 🖥️ Aplicación de monitoreo con Streamlit



El Avance 3 incorpora una aplicación interactiva desarrollada con \*\*Streamlit\*\* dentro de:



```text

src/model\_monitoring.py

```



La aplicación permite visualizar:



\- modo de monitoreo utilizado;

\- cantidad de registros actuales;

\- cantidad de variables monitoreadas;

\- PSI máximo global;

\- cantidad de variables con alerta;

\- tabla de métricas de Data Drift;

\- estado tipo semáforo;

\- recomendaciones;

\- comparación visual entre distribución histórica y actual;

\- evolución temporal del drift;

\- alertas por período;

\- datos actuales junto con los pronósticos del modelo;

\- distribución de la probabilidad estimada de no pagar a tiempo.



También permite cargar opcionalmente un nuevo dataset en formato `.csv`, `.xlsx` o `.xls` para utilizarlo como población actual en reemplazo de la simulación con `TEST`.



\---



\## ⚠️ Interpretación de alertas



El monitoreo distingue entre alertas globales y temporales. Las primeras evalúan si la población actual completa presenta cambios relevantes respecto de la referencia histórica; las segundas evalúan cada período mensual por separado.



Por este motivo, puede existir estabilidad global y, al mismo tiempo, períodos particulares con advertencias o valores críticos.



Una alerta de Data Drift no implica automáticamente reentrenar el modelo. Ante una alerta crítica se recomienda revisar la variable afectada, validar la calidad y origen de los datos, verificar si el cambio persiste, analizar el rendimiento predictivo cuando exista `ground truth` y considerar reentrenamiento solo si la evidencia lo justifica.



\---



\## 🌿 Flujo de trabajo con Git



El repositorio utiliza diferentes ramas para separar desarrollo, integración y versiones estables.



```text

feature/\*

&#x20;   ↓

developer

&#x20;   ↓

main

```



Las ramas `feature/\*` se utilizan para desarrollar funcionalidades específicas. `developer` funciona como rama de integración y `main` contiene las versiones estables y aprobadas.



Los cambios se incorporan mediante Pull Requests para mantener trazabilidad sobre la evolución del proyecto.



\---



\## ⚙️ Instalación



Clonar el repositorio:



```bash

git clone https://github.com/FrancoCorrea90/mlops\_pipeline.git

```



Ingresar al proyecto:



```bash

cd mlops\_pipeline

```



Crear un entorno virtual:



```bash

python -m venv .venv

```



Activarlo en Windows PowerShell:



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



Instalar las dependencias:



```bash

pip install -r requirements.txt

```



\---



\## ▶️ Ejecución



\### Feature Engineering



```bash

python src/ft\_engineering.py

```



\### Entrenamiento y selección del modelo



```bash

python src/model\_train\_evaluation.py

```



Este paso genera localmente `best\_model.joblib`.



\### Aplicación de monitoreo



```bash

python -m streamlit run src/model\_monitoring.py

```



Por defecto, Streamlit inicia la aplicación en una dirección local similar a:



```text

http://localhost:8501

```



\---



\## 🧰 Tecnologías utilizadas



\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- XGBoost

\- SciPy

\- Matplotlib

\- Streamlit

\- Jupyter Notebook

\- Joblib

\- Git

\- GitHub



\---



\## 🔄 Evolución del pipeline



```text

Datos

&#x20; ↓

EDA

&#x20; ↓

Feature Engineering

&#x20; ↓

Entrenamiento

&#x20; ↓

Evaluación

&#x20; ↓

Selección del modelo

&#x20; ↓

Persistencia

&#x20; ↓

Monitoreo de Data Drift

&#x20; ↓

Aplicación Streamlit

&#x20; ↓

Próximas etapas de producción

```

\---



\## 🎓 Contexto



\*\*Proyecto Integrador\*\*  

\*\*Módulo 5 - Producción ML / MLOps\*\*  



El proyecto se desarrolla de manera incremental aplicando conceptos de Machine Learning, ingeniería de características, reproducibilidad, control de versiones, monitoreo y buenas prácticas de MLOps.



