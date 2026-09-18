# 💳 Predicción de Comportamiento de Pago - MLOps Pipeline

Proyecto de Machine Learning orientado a la construcción de un pipeline MLOps para predecir el comportamiento de pago de clientes de una entidad financiera.

El desarrollo integra las etapas de comprensión y preparación de datos, ingeniería de características, entrenamiento y evaluación de modelos, persistencia del modelo, monitoreo de Data Drift, visualización mediante una aplicación desarrollada en Streamlit y disponibilización del modelo mediante una API construida con FastAPI y contenerizada con Docker.

---

## 🎯 1. Caso de negocio

El objetivo del proyecto es desarrollar un modelo de Machine Learning capaz de predecir si un cliente realizará el pago de su crédito en tiempo y forma.

La variable objetivo utilizada es:

```text
Pago_atiempo
```

donde:

- `1` representa clientes que pagan a tiempo.
- `0` representa clientes que no pagan a tiempo.

El problema corresponde a una tarea de **clasificación binaria con clases desbalanceadas**, ya que aproximadamente el 95 % de los registros corresponde a clientes que pagan a tiempo y alrededor del 5 % a clientes que no lo hacen.

Desde el punto de vista del negocio, la detección de la clase minoritaria resulta especialmente relevante, debido a que permite identificar anticipadamente clientes con mayor riesgo de incumplimiento.

---

## 📂 2. Estructura del proyecto

La estructura principal del repositorio es la siguiente:

```text
mlops_pipeline/
│
├── src/
│   ├── Cargar_datos.ipynb
│   ├── comprension_eda.ipynb
│   ├── ft_engineering.py
│   ├── model_train_evaluation.py
│   ├── model_monitoring.py
│   └── model_deploy.py
│
├── Base_de_datos.xlsx
├── best_model.joblib
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

### Principales componentes

- `Cargar_datos.ipynb`: carga y revisión inicial de la base de datos.
- `comprension_eda.ipynb`: análisis exploratorio y comprensión de las variables.
- `ft_engineering.py`: limpieza, Feature Engineering y definición del preprocesamiento.
- `model_train_evaluation.py`: entrenamiento, evaluación, comparación y persistencia de modelos.
- `model_monitoring.py`: análisis de Data Drift y dashboard de monitoreo.
- `model_deploy.py`: disponibilización del modelo mediante una API desarrollada con FastAPI.
- `best_model.joblib`: Pipeline entrenado y persistido para realizar inferencia.
- `requirements.txt`: dependencias necesarias para reproducir el entorno.
- `Dockerfile`: instrucciones para construir la imagen Docker del servicio de inferencia.
- `.dockerignore`: exclusión de archivos innecesarios durante la construcción de la imagen.

---

## 🔎 3. Comprensión y exploración de los datos

El dataset utilizado contiene:

- **10.763 registros**
- **23 variables**
- Sin registros duplicados relevantes.
- Variable objetivo: `Pago_atiempo`.

Durante el análisis exploratorio se estudiaron:

- estructura del dataset;
- tipos de variables;
- valores faltantes;
- distribución de variables numéricas y categóricas;
- distribución de la variable objetivo;
- relaciones entre variables;
- posibles variables con riesgo de fuga de información.

Uno de los principales hallazgos fue el fuerte desbalance de la variable objetivo:

```text
Pago_atiempo = 1 → aproximadamente 95 %
Pago_atiempo = 0 → aproximadamente 5 %
```

Por esta razón, la evaluación del modelo no se basa únicamente en Accuracy, sino que se presta especial atención a métricas relacionadas con la detección de la clase minoritaria.

---

## ⚙️ 4. Ingeniería de características

El procesamiento de datos se encuentra centralizado en:

```text
src/ft_engineering.py
```

Las principales transformaciones realizadas incluyen:

- limpieza y validación de datos;
- conversión de variables categóricas;
- tratamiento de valores faltantes;
- generación de variables temporales;
- separación de variables predictoras y variable objetivo;
- división estratificada entre entrenamiento y prueba;
- construcción del pipeline de preprocesamiento.

A partir de `fecha_prestamo` se generan:

```text
anio_prestamo
mes_prestamo
```

Las variables numéricas utilizan imputación por mediana.

Las variables categóricas utilizan imputación por moda y codificación One-Hot Encoding.

La variable ordinal:

```text
tendencia_ingresos
```

se procesa respetando el siguiente orden:

```text
Decreciente
Estable
Creciente
```

---

## ⚠️ 5. Prevención de Data Leakage

Durante el análisis se identificaron variables que podían contener información disponible posteriormente al otorgamiento del crédito o relacionada directamente con el comportamiento de pago.

Por esta razón se excluyeron del conjunto de variables predictoras:

```text
puntaje
saldo_mora
saldo_total
saldo_principal
saldo_mora_codeudor
```

Esta decisión permite reducir el riesgo de **Data Leakage** y obtener una evaluación más realista del modelo.

---

## ✂️ 6. División de los datos

Se utilizó una división estratificada:

```text
80 % entrenamiento
20 % prueba
```

con:

```text
random_state = 42
```

La distribución resultante fue:

```text
Train: 8.610 registros
Test:  2.153 registros
```

La estratificación permite mantener aproximadamente la misma proporción de clases en ambos conjuntos.

---

## 🤖 7. Entrenamiento y comparación de modelos

El entrenamiento se encuentra implementado en:

```text
src/model_train_evaluation.py
```

Se compararon tres modelos de clasificación:

1. Logistic Regression
2. Random Forest
3. XGBoost

Debido al desbalance de clases, se aplicaron estrategias específicas de ponderación:

- `class_weight="balanced"` en Logistic Regression.
- `class_weight="balanced"` en Random Forest.
- pesos balanceados de muestra para XGBoost.

Las métricas utilizadas fueron:

- Accuracy
- Balanced Accuracy
- Precision de la clase 0
- Recall de la clase 0
- F1-score de la clase 0
- ROC-AUC
- PR-AUC
- Matriz de confusión

---

## 📊 8. Resultados de los modelos

| Modelo | Accuracy | Balanced Accuracy | Precision clase 0 | Recall clase 0 | F1 clase 0 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6521 | 0.5752 | 0.0669 | 0.4902 | 0.1178 | 0.6200 | 0.0786 |
| Random Forest | 0.9526 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.6700 | 0.1441 |
| XGBoost | 0.8402 | 0.5994 | 0.1097 | 0.3333 | **0.1650** | **0.6787** | **0.1485** |

Aunque Random Forest presenta una Accuracy elevada, no logra identificar correctamente observaciones pertenecientes a la clase minoritaria.

El modelo seleccionado fue **XGBoost**, principalmente por presentar el mejor F1-score para la clase `0`, junto con mejores valores de ROC-AUC y PR-AUC.

---

## 💾 9. Persistencia del modelo

Una vez seleccionado el mejor modelo, se almacena utilizando `joblib`:

```text
best_model.joblib
```

Este archivo permite reutilizar el pipeline completo sin necesidad de volver a entrenar el modelo cada vez que se realiza una predicción o un proceso de monitoreo.

El archivo se genera localmente ejecutando:

```bash
python src/model_train_evaluation.py

```
El artefacto `best_model.joblib` contiene el **Pipeline completo utilizado durante el entrenamiento**, integrando:

```text
Preprocesamiento
      ↓
Modelo de clasificación
```

Esto permite reutilizar exactamente las mismas transformaciones aprendidas durante el entrenamiento al momento de realizar nuevas predicciones.

En la etapa de despliegue, `model_deploy.py` carga directamente este Pipeline mediante `joblib`, por lo que no es necesario volver a entrenar el modelo para realizar inferencia.
---

# 📈 10. Monitoreo de Data Drift

El monitoreo del comportamiento de los datos se encuentra implementado en:

```text
src/model_monitoring.py
```

El objetivo es comparar una población histórica de referencia contra una población actual y detectar cambios en la distribución de las variables que puedan afectar el comportamiento del modelo.

En ausencia de datos reales posteriores a la puesta en producción, el proyecto utiliza:

```text
TRAIN → población histórica de referencia
TEST  → población actual simulada
```

Esta configuración permite demostrar la lógica completa de monitoreo sin presentar el conjunto TEST como datos reales de producción.

La aplicación también permite cargar posteriormente un nuevo lote de datos mediante archivos:

```text
CSV
XLSX
XLS
```

sin modificar la lógica principal del sistema.

---

## 📐 11. Métricas de Data Drift

El monitoreo utiliza diferentes métricas según el tipo de variable.

### Variables numéricas

Se utilizan:

- **Population Stability Index (PSI)**
- **Kolmogorov-Smirnov (KS)**
- **Jensen-Shannon Divergence**

### Variables categóricas

Se utilizan:

- **Population Stability Index (PSI)**
- **Chi-cuadrado**
- **Jensen-Shannon Divergence**

La utilización de varias métricas permite evitar que la decisión dependa únicamente de un único indicador.

---

## 🚦 12. Criterio de clasificación del drift

El sistema utiliza tres estados:

```text
🟢 Estable
🟡 Vigilancia
🔴 Drift
```

### PSI

```text
PSI < 0.10          → Estable
0.10 ≤ PSI < 0.25   → Vigilancia
PSI ≥ 0.25          → Drift
```

### Kolmogorov-Smirnov

Para variables numéricas:

```text
KS < 0.15           → Sin señal relevante
0.15 ≤ KS < 0.25    → Vigilancia
KS ≥ 0.25           → Drift
```

El `p-value` de KS también se analiza como evidencia estadística.

Un `p-value < 0.05` por sí solo genera una señal de vigilancia, pero no necesariamente implica un cambio relevante desde el punto de vista práctico.

### Jensen-Shannon

```text
JS < 0.15           → Sin señal relevante
0.15 ≤ JS < 0.30    → Vigilancia
JS ≥ 0.30           → Drift
```

### Chi-cuadrado

Para variables categóricas:

```text
p-value < 0.05
```

indica una diferencia estadísticamente significativa entre las distribuciones.

Sin embargo, Chi-cuadrado no se utiliza de forma aislada para declarar automáticamente un drift crítico.

El resultado se interpreta en conjunto con PSI y Jensen-Shannon.

---

## 🧠 13. Lógica combinada del semáforo

El estado final de cada variable se determina combinando diferentes señales.

Para variables numéricas se consideran:

```text
PSI + KS + Jensen-Shannon
```

Para variables categóricas:

```text
PSI + Chi-cuadrado + Jensen-Shannon
```

Las métricas estadísticas permiten detectar diferencias entre poblaciones, mientras que las métricas de magnitud ayudan a determinar si el cambio observado es suficientemente importante desde el punto de vista operativo.

De esta manera se evita clasificar una variable como crítica únicamente porque una prueba estadística arroje significancia.

---

# 🖥️ 14. Dashboard de monitoreo en Streamlit

El sistema de monitoreo cuenta con una aplicación interactiva desarrollada en **Streamlit**.

La aplicación se organiza en tres secciones principales.

---

## 📊 Tab 1 - Visualización de métricas

Presenta el estado global del lote actual comparando:

```text
Población histórica completa
vs.
Población actual completa
```

Incluye:

- cantidad total de variables monitoreadas;
- variables estables;
- variables en vigilancia;
- variables con drift;
- PSI máximo;
- tabla consolidada de métricas;
- filtros por tipo de variable;
- filtros por estado;
- selección individual de variables;
- comparación gráfica entre distribución histórica y actual;
- interpretación y recomendación de cada variable.

Este análisis representa el **estado global del lote**.

---

## 📅 Tab 2 - Análisis temporal

El monitoreo temporal analiza cada período de forma independiente contra la referencia histórica.

La periodicidad definida es:

```text
Mensual
```

y se requiere un mínimo de:

```text
50 registros por período
```

Las variables:

```text
anio_prestamo
mes_prestamo
```

se utilizan para construir los períodos, pero no se incluyen como variables de drift dentro del análisis temporal para evitar generar desviaciones artificiales.

El dashboard permite visualizar:

- PSI promedio por período;
- PSI máximo;
- cantidad de variables en vigilancia;
- cantidad de variables con drift;
- variable más afectada;
- estado general del período;
- evolución temporal;
- detalle de todas las variables para un mes seleccionado.

El análisis temporal puede detectar desviaciones locales aunque la población global permanezca estable.

Por esta razón, un período puede aparecer en vigilancia o drift aun cuando el estado global del lote sea estable.

Estas diferencias también pueden estar asociadas a factores como estacionalidad, cambios en la composición de la población o tamaños de muestra más pequeños, por lo que deben analizarse considerando su persistencia en el tiempo.

---

## 🚨 Tab 3 - Recomendaciones y alertas

Esta sección centraliza las señales detectadas por el sistema.

Se diferencian:

```text
Alertas globales
Alertas temporales
```

Las alertas globales corresponden al lote actual completo.

Las alertas temporales corresponden a desviaciones detectadas en períodos individuales.

Para cada señal se informa:

- variable afectada;
- tipo de variable;
- estado;
- métricas que originaron la alerta;
- recomendación sugerida.

El sistema también genera notificaciones internas dentro de la aplicación cuando cambia el conjunto de variables en vigilancia o drift.

Además, las alertas pueden exportarse en formato:

```text
CSV
```

para facilitar su análisis o integración posterior con otros procesos.

---

## 🔔 15. Interpretación de las alertas

Las alertas no implican automáticamente que el modelo haya dejado de funcionar correctamente.

El **Data Drift** indica que la distribución de las variables de entrada ha cambiado respecto de la población utilizada como referencia.

Ante una señal persistente de drift se recomienda:

1. verificar calidad e integridad de los datos;
2. analizar cambios en la fuente de información;
3. estudiar si existen cambios en el comportamiento de la población;
4. evaluar las métricas de desempeño del modelo cuando se disponga de la variable objetivo real;
5. considerar un nuevo entrenamiento si el cambio persiste y afecta el rendimiento predictivo.

---

## 📌 16. Principales hallazgos del monitoreo

La comparación global entre TRAIN y TEST muestra una población relativamente estable, lo cual resulta esperable debido a que ambos conjuntos pertenecen al mismo universo de datos.

Sin embargo, el análisis temporal permite detectar períodos individuales con desviaciones más importantes.

Esto demuestra una diferencia relevante entre:

```text
Estabilidad global
vs.
Desviaciones locales por período
```

El monitoreo temporal permite identificar cambios que pueden quedar ocultos al observar únicamente la distribución agregada.

---

# 🔄 17. Flujo general del pipeline

El flujo implementado puede resumirse de la siguiente manera:

```text
Datos originales
      ↓
Comprensión y EDA
      ↓
Limpieza de datos
      ↓
Ingeniería de características
      ↓
Prevención de Data Leakage
      ↓
Train / Test Split
      ↓
Preprocesamiento
      ↓
Entrenamiento de modelos
      ↓
Evaluación
      ↓
Selección de XGBoost
      ↓
Persistencia del Pipeline
      ↓
Monitoreo de Data Drift
      ↓
Análisis temporal
      ↓
Alertas y recomendaciones
      ↓
Dashboard Streamlit
      ↓
API con FastAPI
      ↓
Predicción individual / batch
      ↓
Contenerización con Docker

---

# ▶️ 18. Instalación

Clonar el repositorio:

```bash
git clone https://github.com/FrancoCorrea90/mlops_pipeline.git
```

Ingresar al proyecto:

```bash
cd mlops_pipeline
```

Crear y activar un entorno virtual.

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

# 🤖 19. Entrenamiento del modelo

Ejecutar:

```bash
python src/model_train_evaluation.py
```

Este proceso entrena los modelos, compara las métricas, selecciona el mejor modelo y genera localmente:

```text
best_model.joblib
```

---

# 📊 20. Ejecución del dashboard

Desde la raíz del proyecto:

```bash
python -m streamlit run src/model_monitoring.py
```

Si la terminal se encuentra dentro de `src`:

```bash
python -m streamlit run model_monitoring.py
```

Streamlit iniciará un servidor local y permitirá acceder al dashboard desde el navegador.

Generalmente:

```text
http://localhost:8501
```

---

# 🛠️ 21. Tecnologías utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SciPy
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- Jupyter Notebook
- OpenPyXL
- FastAPI
- Pydantic
- Uvicorn
- Docker
- Git
- GitHub

---

# 🌿 22. Flujo de trabajo con Git

El proyecto utiliza una estructura de ramas para mantener separado el desarrollo de nuevas funcionalidades del código estable.

Las ramas principales son:

```text
main
developer
certification
```

Las nuevas funcionalidades se desarrollan en ramas temporales:

```text
feature/*
```

La rama developer se utiliza para integrar las funcionalidades desarrolladas en ramas feature/*.

Una vez consolidado el desarrollo, los cambios se promueven a certification, donde se valida el funcionamiento integral del proyecto antes de generar la versión final estable en main.

El flujo general utilizado es:

```text
feature/*
    ↓
developer
    ↓
certification
    ↓
main
```

Los cambios se integran mediante Pull Requests y las ramas temporales se eliminan una vez finalizada su integración.

---

# 🚀 23. API de predicción con FastAPI

El modelo entrenado se disponibiliza mediante una API desarrollada con **FastAPI**, implementada en:

```text
src/model_deploy.py
```

Al iniciar la aplicación se carga:

```text
best_model.joblib
```

Este archivo contiene el Pipeline completo de Machine Learning utilizado durante el entrenamiento, compuesto por:

```text
Preprocesamiento
      ↓
Modelo de clasificación
```

De esta manera, el modelo puede realizar nuevas predicciones sin necesidad de volver a ejecutar el proceso de entrenamiento.

La API cuenta con tres endpoints principales:

| Método | Endpoint | Descripción |
|---|---|---|
| GET | `/health` | Verifica que la API esté activa y que el modelo haya sido cargado correctamente. |
| POST | `/predict` | Realiza una predicción para un único registro. |
| POST | `/predict/batch` | Permite enviar múltiples registros en una sola solicitud y obtener predicciones por lotes. |

## Preparación de los datos para inferencia

Durante la etapa de Feature Engineering, la variable original:

```text
fecha_prestamo
```

fue transformada en:

```text
anio_prestamo
mes_prestamo
```

Estas variables son utilizadas posteriormente por el modelo.

Por esta razón, la API recibe `fecha_prestamo` y reproduce esta transformación antes de realizar la inferencia:

```text
fecha_prestamo
      ↓
anio_prestamo
mes_prestamo
      ↓
Pipeline
      ↓
Predicción
```

Asimismo, `tipo_credito` se transforma a texto para mantener consistencia con el tratamiento realizado durante el entrenamiento.

Las restantes transformaciones se encuentran incorporadas dentro del Pipeline persistido:

- imputación de variables numéricas mediante mediana;
- imputación de variables categóricas mediante el valor más frecuente;
- One-Hot Encoding para variables categóricas nominales;
- codificación ordinal para `tendencia_ingresos`.

## Validación de entradas

La API utiliza **Pydantic** para definir y validar el esquema de los datos recibidos.

Esto permite controlar los tipos de las variables antes de que la información sea enviada al modelo.

Entre las variables de entrada se encuentran:

```text
tipo_credito
capital_prestado
plazo_meses
edad_cliente
tipo_laboral
salario_cliente
total_otros_prestamos
cuota_pactada
puntaje_datacredito
cant_creditosvigentes
huella_consulta
creditos_sectorFinanciero
creditos_sectorCooperativo
creditos_sectorReal
promedio_ingresos_datacredito
tendencia_ingresos
fecha_prestamo
```

## Ejecución local de la API

Desde el directorio `src`:

```bash
python -m uvicorn model_deploy:app --reload
```

Una vez iniciado el servidor, la documentación interactiva generada automáticamente por FastAPI puede consultarse mediante Swagger UI:

```text
http://127.0.0.1:8000/docs
```

El estado del servicio puede verificarse mediante:

```text
http://127.0.0.1:8000/health
```

Una respuesta válida es:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

## Predicción individual

El endpoint:

```text
POST /predict
```

recibe un único registro y devuelve la clase predicha junto con una interpretación del resultado.

Ejemplo:

```json
{
  "prediction": 0,
  "resultado": "No paga a tiempo"
}
```

La interpretación utilizada es:

```text
0 → No paga a tiempo
1 → Paga a tiempo
```

## Predicción por lotes

El endpoint:

```text
POST /predict/batch
```

permite enviar múltiples registros en una única solicitud.

Los registros son transformados en un DataFrame y procesados conjuntamente por el modelo.

Ejemplo de respuesta:

```json
{
  "cantidad_registros": 2,
  "predictions": [
    {
      "registro": 1,
      "prediction": 0,
      "resultado": "No paga a tiempo"
    },
    {
      "registro": 2,
      "prediction": 1,
      "resultado": "Paga a tiempo"
    }
  ]
}
```

---

# 🐳 24. Contenerización con Docker

Con el objetivo de disponer de un entorno reproducible para ejecutar el servicio de inferencia, la API fue contenerizada mediante **Docker**.

La imagen utiliza:

```text
Python 3.12
```

e incorpora:

```text
Código fuente
      +
Dependencias
      +
Pipeline entrenado
      +
FastAPI
      +
Uvicorn
```

El proceso de construcción se encuentra definido en:

```text
Dockerfile
```

## Dockerfile

La imagen parte de:

```dockerfile
FROM python:3.12-slim
```

y utiliza:

```text
/app
```

como directorio de trabajo dentro del contenedor.

El flujo de construcción puede resumirse como:

```text
Imagen Python 3.12
      ↓
Copiar requirements.txt
      ↓
Instalar dependencias
      ↓
Copiar best_model.joblib
      ↓
Copiar src/
      ↓
Exponer puerto 8000
      ↓
Iniciar Uvicorn
```

La aplicación se ejecuta mediante:

```text
uvicorn src.model_deploy:app
```

escuchando conexiones en:

```text
0.0.0.0:8000
```

## Construcción de la imagen

Desde la raíz del repositorio:

```bash
docker build -t mlops-pipeline-api:1.0 .
```

La imagen generada utiliza:

```text
Nombre: mlops-pipeline-api
Tag:    1.0
```

## Ejecución del contenedor

El contenedor puede iniciarse mediante:

```bash
docker run --name mlops-pipeline-container -p 8000:8000 mlops-pipeline-api:1.0
```

El parámetro:

```text
-p 8000:8000
```

realiza el siguiente mapeo:

```text
Puerto 8000 del equipo host
            ↓
Puerto 8000 del contenedor
            ↓
Uvicorn
            ↓
FastAPI
```

Una vez iniciado el contenedor, la documentación de la API puede consultarse desde el equipo host en:

```text
http://127.0.0.1:8000/docs
```

## .dockerignore

El archivo:

```text
.dockerignore
```

permite excluir del contexto de construcción elementos que no son necesarios para ejecutar el servicio, como:

- entornos virtuales;
- caché de Python;
- archivos temporales;
- configuración local del editor;
- archivos de Git;
- documentación local.

Esto permite mantener el contexto de construcción más limpio y evitar incorporar archivos innecesarios a la imagen.

---

# ✅ 25. Validación del despliegue

El servicio fue validado tanto en ejecución local como dentro de un contenedor Docker.

Se verificaron exitosamente:

- carga de `best_model.joblib`;
- inicialización de FastAPI;
- ejecución del servicio mediante Uvicorn;
- endpoint `GET /health`;
- predicción individual mediante `POST /predict`;
- predicción por lotes mediante `POST /predict/batch`;
- validación automática de entradas mediante Pydantic;
- transformación de `fecha_prestamo` en año y mes;
- documentación interactiva mediante Swagger UI;
- construcción de la imagen Docker;
- creación y ejecución del contenedor;
- acceso a la API desde el equipo host.

Las pruebas realizadas sobre los endpoints de predicción devolvieron:

```text
HTTP 200 OK
```

confirmando que el Pipeline puede realizar inferencia correctamente dentro del entorno contenerizado.

El flujo final de inferencia es:

```text
Solicitud HTTP
      ↓
FastAPI
      ↓
Validación con Pydantic
      ↓
Feature Engineering de inferencia
      ↓
DataFrame
      ↓
best_model.joblib
      ↓
Pipeline
      ↓
Preprocesamiento
      ↓
Modelo de Machine Learning
      ↓
Predicción
      ↓
Respuesta JSON
```

Además, la versión integrada del proyecto fue promovida mediante el siguiente flujo de ramas:

```text
feature/*
    ↓
developer
    ↓
certification
    ↓
main
```

La rama `certification` se utilizó como instancia de validación integral previa a la promoción de la versión estable a `main`.

---

# 📌 26. Estado actual del proyecto

Actualmente el proyecto integra las principales etapas del ciclo de vida de una solución de Machine Learning:

- carga y comprensión de los datos;
- análisis exploratorio;
- ingeniería de características;
- prevención de Data Leakage;
- división estratificada de entrenamiento y prueba;
- construcción de pipelines de preprocesamiento;
- entrenamiento y comparación de diferentes algoritmos;
- selección del modelo con mejor desempeño para la clase minoritaria;
- persistencia del Pipeline entrenado;
- monitoreo de Data Drift;
- análisis global y temporal de cambios en las distribuciones;
- generación de alertas y recomendaciones;
- visualización mediante un dashboard interactivo en Streamlit;
- disponibilización del modelo mediante una API desarrollada con FastAPI;
- validación automática de datos de entrada mediante Pydantic;
- inferencia para registros individuales;
- inferencia por lotes;
- ejecución de la API mediante Uvicorn;
- contenerización del servicio mediante Docker;
- validación integral previa a la publicación de la versión estable.

El flujo general alcanzado es:

```text
Comprensión de datos
        ↓
Feature Engineering
        ↓
Entrenamiento
        ↓
Evaluación
        ↓
Persistencia del Pipeline
        ↓
Monitoreo de Data Drift
        ↓
Dashboard de monitoreo
        ↓
API de inferencia
        ↓
Predicción individual y batch
        ↓
Contenerización con Docker
```

El Pipeline entrenado se encuentra persistido mediante `joblib` y puede utilizarse para realizar nuevas predicciones sin necesidad de volver a ejecutar el entrenamiento.

El dashboard desarrollado con Streamlit permite analizar métricas de Data Drift, evolución temporal y recomendaciones asociadas a posibles cambios en la distribución de los datos.

La API desarrollada con FastAPI permite disponibilizar el modelo para realizar inferencia mediante solicitudes HTTP, tanto para registros individuales como para múltiples observaciones.

Finalmente, Docker permite ejecutar el servicio de inferencia dentro de un entorno reproducible que contiene las dependencias, el código fuente y el modelo necesarios para su funcionamiento.

De esta manera, el proyecto integra preparación de datos, entrenamiento, evaluación, persistencia, monitoreo y despliegue dentro de un mismo pipeline de Machine Learning.

---