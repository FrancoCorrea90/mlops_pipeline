from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# ==================================================
# CONFIGURACIÓN
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "best_model.joblib"


# ==================================================
# CARGA DEL MODELO
# ==================================================

model = joblib.load(MODEL_PATH)


# ==================================================
# ESQUEMA DE DATOS DE ENTRADA
# ==================================================

class PredictionInput(BaseModel):
    tipo_credito: int
    capital_prestado: float
    plazo_meses: int
    edad_cliente: int
    tipo_laboral: str
    salario_cliente: float
    total_otros_prestamos: float
    cuota_pactada: float
    puntaje_datacredito: float
    cant_creditosvigentes: int
    huella_consulta: int
    creditos_sectorFinanciero: int
    creditos_sectorCooperativo: int
    creditos_sectorReal: int
    promedio_ingresos_datacredito: float
    tendencia_ingresos: str
    fecha_prestamo: datetime


# ==================================================
# API
# ==================================================

app = FastAPI(
    title="API de Predicción de Pago a Tiempo",
    description=(
        "API para realizar predicciones utilizando "
        "el modelo de Machine Learning entrenado."
    ),
    version="1.0.0",
)


# ==================================================
# ENDPOINT DE ESTADO
# ==================================================

@app.get("/health")
def health_check():
    """
    Verifica que la API esté funcionando
    y que el modelo haya sido cargado.
    """
    return {
        "status": "ok",
        "model_loaded": True,
    }


# ==================================================
# ENDPOINT DE PREDICCIÓN
# ==================================================

@app.post("/predict")
def predict(data: PredictionInput):
    """
    Recibe los datos de un cliente,
    realiza el feature engineering necesario
    y devuelve la predicción del modelo.
    """

    # Convertimos los datos recibidos a diccionario.
    input_data = data.model_dump()

    # Replicamos el tratamiento realizado durante
    # el feature engineering del entrenamiento.
    input_data["tipo_credito"] = str(
        input_data["tipo_credito"]
    )

    fecha = input_data.pop("fecha_prestamo")

    input_data["anio_prestamo"] = fecha.year
    input_data["mes_prestamo"] = fecha.month

    # Convertimos el registro a DataFrame.
    input_df = pd.DataFrame([input_data])

    # Realizamos la predicción.
    prediction = int(
        model.predict(input_df)[0]
    )

    # Interpretación de la clase predicha.
    if prediction == 1:
        resultado = "Paga a tiempo"
    else:
        resultado = "No paga a tiempo"

    return {
        "prediction": prediction,
        "resultado": resultado,
    }

# ==================================================
# ENDPOINT DE PREDICCIÓN POR LOTES
# ==================================================

@app.post("/predict/batch")
def predict_batch(data: list[PredictionInput]):
    """
    Recibe múltiples registros en una sola solicitud
    y devuelve una predicción para cada uno.
    """

    processed_records = []

    for record in data:

        input_data = record.model_dump()

        # Mismo feature engineering utilizado
        # para la predicción individual.
        input_data["tipo_credito"] = str(
            input_data["tipo_credito"]
        )

        fecha = input_data.pop("fecha_prestamo")

        input_data["anio_prestamo"] = fecha.year
        input_data["mes_prestamo"] = fecha.month

        processed_records.append(input_data)

    # Convertimos todos los registros juntos
    # a un único DataFrame.
    input_df = pd.DataFrame(processed_records)

    # El modelo realiza todas las predicciones
    # en una única llamada.
    predictions = model.predict(input_df)

    results = []

    for index, prediction in enumerate(predictions):

        prediction = int(prediction)

        if prediction == 1:
            resultado = "Paga a tiempo"
        else:
            resultado = "No paga a tiempo"

        results.append(
            {
                "registro": index + 1,
                "prediction": prediction,
                "resultado": resultado,
            }
        )

    return {
        "cantidad_registros": len(results),
        "predictions": results,
    }