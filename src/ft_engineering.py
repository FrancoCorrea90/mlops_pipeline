
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Base_de_datos.xlsx"

TARGET = "Pago_atiempo"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# VARIABLES CON POSIBLE DATA LEAKAGE
# ============================================================

# Ante la ausencia de un diccionario de datos, estas variables
# se excluyen del modelo principal por no poder garantizar que
# estén disponibles al momento de otorgar el crédito.
POTENTIAL_LEAKAGE_FEATURES = [
    "saldo_mora",
    "saldo_total",
    "saldo_principal",
    "saldo_mora_codeudor",
]


# ============================================================
# DEFINICIÓN DE VARIABLES
# ============================================================

NUMERIC_FEATURES = [
    "capital_prestado",
    "plazo_meses",
    "edad_cliente",
    "salario_cliente",
    "total_otros_prestamos",
    "cuota_pactada",
    "puntaje",
    "puntaje_datacredito",
    "cant_creditosvigentes",
    "huella_consulta",
    "creditos_sectorFinanciero",
    "creditos_sectorCooperativo",
    "creditos_sectorReal",
    "promedio_ingresos_datacredito",
    "anio_prestamo",
    "mes_prestamo",
]

CATEGORICAL_FEATURES = [
    "tipo_credito",
    "tipo_laboral",
]

ORDINAL_FEATURES = [
    "tendencia_ingresos",
]

TENDENCIA_CATEGORIES = [
    "Decreciente",
    "Estable",
    "Creciente",
]


# ============================================================
# CARGA DE DATOS
# ============================================================

def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Carga el dataset original desde un archivo Excel.
    """
    return pd.read_excel(path)


# ============================================================
# LIMPIEZA Y FEATURE ENGINEERING
# ============================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza limpieza inicial y feature engineering antes
    de dividir los datos en entrenamiento y prueba.
    """

    df = df.copy()

    # --------------------------------------------------------
    # 1. Validación de la variable objetivo
    # --------------------------------------------------------

    if TARGET not in df.columns:
        raise ValueError(
            f"No se encontró la variable objetivo '{TARGET}'."
        )

    # --------------------------------------------------------
    # 2. tipo_credito se trata como categórica
    # --------------------------------------------------------

    df["tipo_credito"] = df["tipo_credito"].astype(str)

    # --------------------------------------------------------
    # 3. Limpieza de tendencia_ingresos
    # --------------------------------------------------------

    valid_tendencia = [
        "Decreciente",
        "Estable",
        "Creciente",
    ]

    df.loc[
        ~df["tendencia_ingresos"].isin(valid_tendencia),
        "tendencia_ingresos"
    ] = np.nan

    # --------------------------------------------------------
    # 4. Feature engineering de fecha
    # --------------------------------------------------------

    df["fecha_prestamo"] = pd.to_datetime(
        df["fecha_prestamo"],
        errors="coerce"
    )

    df["anio_prestamo"] = df["fecha_prestamo"].dt.year
    df["mes_prestamo"] = df["fecha_prestamo"].dt.month

    # Ya no necesitamos la fecha original- Extraemos lo importante para el modelo.
    df = df.drop(columns=["fecha_prestamo"])

    # --------------------------------------------------------
    # 5. Eliminación preventiva de posibles variables leakage
    # --------------------------------------------------------

    df = df.drop(
        columns=POTENTIAL_LEAKAGE_FEATURES,
        errors="ignore"
    )

    return df


# ============================================================
# SEPARACIÓN X / y
# ============================================================

def split_features_target(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separa las variables predictoras de la variable objetivo.
    """

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return X, y


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series
):
    """
    Divide los datos en entrenamiento y prueba manteniendo
    la proporción de clases de la variable objetivo.
    """

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


# ============================================================
# PIPELINES DE PREPROCESAMIENTO
# ============================================================

def build_preprocessor() -> ColumnTransformer:
    """
    Construye el ColumnTransformer para variables numéricas,
    categóricas nominales y categóricas ordinales.
    """

    # --------------------------------------------------------
    # Variables numéricas
    # --------------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
        ]
    )

    # --------------------------------------------------------
    # Variables categóricas nominales
    # --------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            ),
        ]
    )

    # --------------------------------------------------------
    # Variable categórica ordinal
    # --------------------------------------------------------

    ordinal_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[
                        TENDENCIA_CATEGORIES
                    ],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )
            ),
        ]
    )

    # --------------------------------------------------------
    # ColumnTransformer
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            ),
            (
                "ordinal",
                ordinal_pipeline,
                ORDINAL_FEATURES
            ),
        ],
        remainder="drop",
    )

    return preprocessor


# ============================================================
# EJECUCIÓN DE PRUEBA
# ============================================================

if __name__ == "__main__":

    # 1. Cargar datos
    data = load_data()

    print("\n===== DATASET ORIGINAL =====")
    print("Shape:", data.shape)

    # 2. Limpiar y crear nuevas variables
    data_clean = clean_data(data)

    print("\n===== DATASET DESPUÉS DE LIMPIEZA =====")
    print("Shape:", data_clean.shape)

    print("\nColumnas:")
    print(data_clean.columns.tolist())

    # 3. Separar X / y
    X, y = split_features_target(data_clean)

    # 4. Train / Test
    X_train, X_test, y_train, y_test = create_train_test_split(
        X,
        y
    )

    print("\n===== TRAIN / TEST =====")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

    print("\n===== DISTRIBUCIÓN TARGET TRAIN =====")
    print(
        y_train
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\n===== DISTRIBUCIÓN TARGET TEST =====")
    print(
        y_test
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    # 5. Crear preprocesador
    preprocessor = build_preprocessor()

    # Se ajusta solamente con TRAIN para evitar data leakage
    X_train_processed = preprocessor.fit_transform(X_train)

    # TEST solamente se transforma
    X_test_processed = preprocessor.transform(X_test)

    print("\n===== PREPROCESAMIENTO =====")
    print(
        "Shape train transformado:",
        X_train_processed.shape
    )

    print(
        "Shape test transformado:",
        X_test_processed.shape
    )

    # 6. Validación final
    print("\n===== VALIDACIÓN FINAL =====")

    print(
        "Nulos en train procesado:",
        pd.DataFrame(X_train_processed).isnull().sum().sum()
    )

    print(
        "Nulos en test procesado:",
        pd.DataFrame(X_test_processed).isnull().sum().sum()
    )

    feature_names = preprocessor.get_feature_names_out()

    print("\nCantidad de features finales:")
    print(len(feature_names))

    print("\nFeatures generadas:")

    for feature in feature_names:
        print(feature)

    print("\nFeature engineering ejecutado correctamente.")