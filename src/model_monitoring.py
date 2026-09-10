from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp

from ft_engineering import (
    BASE_DIR,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    TARGET,
    clean_data,
    create_train_test_split,
    load_data,
    split_features_target,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

MODEL_PATH = BASE_DIR / "best_model.joblib"

EPSILON = 1e-6

PSI_WARNING = 0.10
PSI_CRITICAL = 0.25

MONITORING_FREQUENCY = "Mensual"
MIN_TEMPORAL_SAMPLES = 50

TEMPORAL_FEATURES = [
    "anio_prestamo",
    "mes_prestamo",
]


# ============================================================
# UTILIDADES
# ============================================================

def _safe_distribution(counts: np.ndarray) -> np.ndarray:
    """Convierte conteos en proporciones evitando ceros."""
    counts = np.asarray(counts, dtype=float) + EPSILON
    return counts / counts.sum()


def _reference_bins(
    reference: pd.Series,
    bins: int = 10,
) -> np.ndarray:
    """
    Construye intervalos a partir de cuantiles de la población
    histórica. Los mismos intervalos se reutilizan sobre la actual.
    """
    reference = pd.to_numeric(
        reference,
        errors="coerce",
    ).dropna()

    if reference.empty:
        return np.array([-np.inf, np.inf])

    quantiles = np.linspace(0, 1, bins + 1)

    edges = np.unique(
        reference.quantile(
            quantiles
        ).to_numpy(dtype=float)
    )

    if len(edges) < 2:
        value = float(reference.iloc[0])

        return np.array(
            [-np.inf, value, np.inf]
        )

    edges[0] = -np.inf
    edges[-1] = np.inf

    return edges


# ============================================================
# PSI
# ============================================================

def population_stability_index_numeric(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> float:
    """Calcula PSI para una variable numérica."""
    edges = _reference_bins(
        reference,
        bins=bins,
    )

    ref_values = pd.to_numeric(
        reference,
        errors="coerce",
    ).dropna()

    cur_values = pd.to_numeric(
        current,
        errors="coerce",
    ).dropna()

    ref_counts, _ = np.histogram(
        ref_values,
        bins=edges,
    )

    cur_counts, _ = np.histogram(
        cur_values,
        bins=edges,
    )

    ref_dist = _safe_distribution(
        ref_counts
    )

    cur_dist = _safe_distribution(
        cur_counts
    )

    psi = np.sum(
        (cur_dist - ref_dist)
        * np.log(
            cur_dist / ref_dist
        )
    )

    return float(psi)


def population_stability_index_categorical(
    reference: pd.Series,
    current: pd.Series,
) -> float:
    """Calcula PSI para una variable categórica."""
    ref = (
        reference
        .fillna("MISSING")
        .astype(str)
    )

    cur = (
        current
        .fillna("MISSING")
        .astype(str)
    )

    categories = sorted(
        set(ref.unique())
        | set(cur.unique())
    )

    ref_counts = (
        ref
        .value_counts()
        .reindex(
            categories,
            fill_value=0,
        )
        .to_numpy()
    )

    cur_counts = (
        cur
        .value_counts()
        .reindex(
            categories,
            fill_value=0,
        )
        .to_numpy()
    )

    ref_dist = _safe_distribution(
        ref_counts
    )

    cur_dist = _safe_distribution(
        cur_counts
    )

    psi = np.sum(
        (cur_dist - ref_dist)
        * np.log(
            cur_dist / ref_dist
        )
    )

    return float(psi)


# ============================================================
# JENSEN-SHANNON
# ============================================================

def jensen_shannon_numeric(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> float:
    """Calcula divergencia Jensen-Shannon para una variable numérica."""
    edges = _reference_bins(
        reference,
        bins=bins,
    )

    ref_values = pd.to_numeric(
        reference,
        errors="coerce",
    ).dropna()

    cur_values = pd.to_numeric(
        current,
        errors="coerce",
    ).dropna()

    ref_counts, _ = np.histogram(
        ref_values,
        bins=edges,
    )

    cur_counts, _ = np.histogram(
        cur_values,
        bins=edges,
    )

    ref_dist = _safe_distribution(
        ref_counts
    )

    cur_dist = _safe_distribution(
        cur_counts
    )

    js_divergence = (
        jensenshannon(
            ref_dist,
            cur_dist,
            base=2,
        )
        ** 2
    )

    return float(
        js_divergence
    )


def jensen_shannon_categorical(
    reference: pd.Series,
    current: pd.Series,
) -> float:
    """Calcula divergencia Jensen-Shannon para una variable categórica."""
    ref = (
        reference
        .fillna("MISSING")
        .astype(str)
    )

    cur = (
        current
        .fillna("MISSING")
        .astype(str)
    )

    categories = sorted(
        set(ref.unique())
        | set(cur.unique())
    )

    ref_counts = (
        ref
        .value_counts()
        .reindex(
            categories,
            fill_value=0,
        )
        .to_numpy()
    )

    cur_counts = (
        cur
        .value_counts()
        .reindex(
            categories,
            fill_value=0,
        )
        .to_numpy()
    )

    ref_dist = _safe_distribution(
        ref_counts
    )

    cur_dist = _safe_distribution(
        cur_counts
    )

    js_divergence = (
        jensenshannon(
            ref_dist,
            cur_dist,
            base=2,
        )
        ** 2
    )

    return float(
        js_divergence
    )


# ============================================================
# CLASIFICACIÓN Y RECOMENDACIONES
# ============================================================

def classify_drift(
    psi: float,
) -> str:
    """
    Semáforo operativo basado en PSI.

    PSI < 0.10:
        Estable.

    0.10 <= PSI < 0.25:
        Advertencia.

    PSI >= 0.25:
        Crítico.
    """
    if pd.isna(psi):
        return "Sin datos"

    if psi >= PSI_CRITICAL:
        return "Crítico"

    if psi >= PSI_WARNING:
        return "Advertencia"

    return "Estable"


def drift_recommendation(
    status: str,
) -> str:
    """Devuelve una recomendación operativa según el nivel de drift."""
    if status == "Crítico":
        return (
            "Revisar la variable, validar calidad/origen del dato "
            "y evaluar reentrenamiento si el cambio persiste."
        )

    if status == "Advertencia":
        return (
            "Mantener seguimiento en los próximos períodos "
            "y revisar la tendencia de la variable."
        )

    if status == "Estable":
        return (
            "Sin acción inmediata; continuar monitoreo periódico."
        )

    return (
        "No hay información suficiente para recomendar una acción."
    )


# ============================================================
# MONITOREO POR VARIABLE
# ============================================================

def monitor_numeric_feature(
    reference: pd.Series,
    current: pd.Series,
    feature: str,
) -> dict:
    """Calcula KS, PSI y Jensen-Shannon para una variable numérica."""
    ref = pd.to_numeric(
        reference,
        errors="coerce",
    ).dropna()

    cur = pd.to_numeric(
        current,
        errors="coerce",
    ).dropna()

    if ref.empty or cur.empty:
        return {
            "Variable": feature,
            "Tipo": "Numérica",
            "KS statistic": np.nan,
            "KS p-value": np.nan,
            "PSI": np.nan,
            "Jensen-Shannon": np.nan,
            "Estado": "Sin datos",
            "Recomendación": (
                "No hay información suficiente para evaluar la variable."
            ),
        }

    ks_result = ks_2samp(
        ref,
        cur,
    )

    psi = population_stability_index_numeric(
        ref,
        cur,
    )

    js = jensen_shannon_numeric(
        ref,
        cur,
    )

    status = classify_drift(
        psi
    )

    return {
        "Variable": feature,
        "Tipo": "Numérica",
        "KS statistic": float(
            ks_result.statistic
        ),
        "KS p-value": float(
            ks_result.pvalue
        ),
        "PSI": psi,
        "Jensen-Shannon": js,
        "Estado": status,
        "Recomendación": drift_recommendation(
            status
        ),
    }


def monitor_categorical_feature(
    reference: pd.Series,
    current: pd.Series,
    feature: str,
) -> dict:
    """Calcula Chi-cuadrado, PSI y Jensen-Shannon para una categórica."""
    ref = (
        reference
        .fillna("MISSING")
        .astype(str)
    )

    cur = (
        current
        .fillna("MISSING")
        .astype(str)
    )

    categories = sorted(
        set(ref.unique())
        | set(cur.unique())
    )

    ref_counts = (
        ref
        .value_counts()
        .reindex(
            categories,
            fill_value=0,
        )
    )

    cur_counts = (
        cur
        .value_counts()
        .reindex(
            categories,
            fill_value=0,
        )
    )

    contingency = np.vstack(
        [
            ref_counts.to_numpy(),
            cur_counts.to_numpy(),
        ]
    )

    if len(categories) <= 1:
        chi2_stat = np.nan
        chi2_pvalue = np.nan

    else:
        (
            chi2_stat,
            chi2_pvalue,
            _,
            _,
        ) = chi2_contingency(
            contingency
        )

    psi = population_stability_index_categorical(
        ref,
        cur,
    )

    js = jensen_shannon_categorical(
        ref,
        cur,
    )

    status = classify_drift(
        psi
    )

    return {
        "Variable": feature,
        "Tipo": "Categórica",
        "Chi2 statistic": (
            float(chi2_stat)
            if pd.notna(
                chi2_stat
            )
            else np.nan
        ),
        "Chi2 p-value": (
            float(chi2_pvalue)
            if pd.notna(
                chi2_pvalue
            )
            else np.nan
        ),
        "PSI": psi,
        "Jensen-Shannon": js,
        "Estado": status,
        "Recomendación": drift_recommendation(
            status
        ),
    }


def calculate_drift_table(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> pd.DataFrame:
    """Genera una tabla consolidada de métricas de data drift."""
    rows = []

    for feature in NUMERIC_FEATURES:
        if (
            feature in reference_df.columns
            and feature in current_df.columns
        ):
            rows.append(
                monitor_numeric_feature(
                    reference_df[feature],
                    current_df[feature],
                    feature,
                )
            )

    categorical_features = (
        CATEGORICAL_FEATURES
        + ORDINAL_FEATURES
    )

    for feature in categorical_features:
        if (
            feature in reference_df.columns
            and feature in current_df.columns
        ):
            rows.append(
                monitor_categorical_feature(
                    reference_df[feature],
                    current_df[feature],
                    feature,
                )
            )

    drift_df = pd.DataFrame(
        rows
    )

    if not drift_df.empty:
        drift_df["Alerta"] = (
            drift_df["Estado"]
            .isin(
                [
                    "Advertencia",
                    "Crítico",
                ]
            )
        )

        drift_df = (
            drift_df
            .sort_values(
                by="PSI",
                ascending=False,
                na_position="last",
            )
            .reset_index(
                drop=True
            )
        )

    return drift_df


# ============================================================
# PREDICCIONES
# ============================================================

def build_prediction_table(
    model,
    X_current: pd.DataFrame,
    y_current: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Devuelve los datos actuales junto con el pronóstico del modelo,
    la probabilidad de no pagar a tiempo y, si está disponible,
    el valor real.
    """
    output = X_current.copy()

    predictions = model.predict(
        X_current
    )

    output[
        "prediccion_Pago_atiempo"
    ] = predictions

    if hasattr(
        model,
        "predict_proba",
    ):
        probabilities = model.predict_proba(
            X_current
        )

        classes = (
            model
            .named_steps["model"]
            .classes_
        )

        if 0 in classes:
            class_0_index = (
                list(classes)
                .index(0)
            )

            output[
                "probabilidad_no_pago"
            ] = probabilities[
                :,
                class_0_index,
            ]

    if y_current is not None:
        output[
            "Pago_atiempo_real"
        ] = y_current.to_numpy()

    return output


# ============================================================
# MONITOREO TEMPORAL
# ============================================================

def calculate_temporal_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula la evolución mensual del drift usando una referencia fija.

    anio_prestamo y mes_prestamo se utilizan para construir el período,
    pero se excluyen del cálculo mensual para evitar drift artificial.

    Los períodos con menos de MIN_TEMPORAL_SAMPLES observaciones
    no se informan.
    """
    required = {
        "anio_prestamo",
        "mes_prestamo",
    }

    if not required.issubset(
        current_df.columns
    ):
        return pd.DataFrame()

    temp = current_df.copy()

    temp["periodo"] = pd.to_datetime(
        dict(
            year=temp[
                "anio_prestamo"
            ],
            month=temp[
                "mes_prestamo"
            ],
            day=1,
        ),
        errors="coerce",
    )

    reference_features = (
        reference_df
        .drop(
            columns=TEMPORAL_FEATURES,
            errors="ignore",
        )
    )

    rows = []

    grouped = (
        temp
        .dropna(
            subset=["periodo"]
        )
        .groupby("periodo")
    )

    for period, period_df in grouped:
        sample_size = len(
            period_df
        )

        if (
            sample_size
            < MIN_TEMPORAL_SAMPLES
        ):
            continue

        period_features = (
            period_df
            .drop(
                columns=(
                    TEMPORAL_FEATURES
                    + ["periodo"]
                ),
                errors="ignore",
            )
        )

        drift = calculate_drift_table(
            reference_df=reference_features,
            current_df=period_features,
        )

        if drift.empty:
            continue

        valid_psi = drift.dropna(
            subset=["PSI"]
        )

        if valid_psi.empty:
            continue

        max_index = (
            valid_psi["PSI"]
            .idxmax()
        )

        most_affected = (
            valid_psi
            .loc[max_index]
        )

        max_psi = float(
            most_affected[
                "PSI"
            ]
        )

        period_status = classify_drift(
            max_psi
        )

        rows.append(
            {
                "Periodo": period,
                "Muestras actuales": sample_size,
                "PSI promedio": float(
                    valid_psi[
                        "PSI"
                    ].mean()
                ),
                "PSI máximo": max_psi,
                "Variable más afectada": (
                    most_affected[
                        "Variable"
                    ]
                ),
                "Estado período": period_status,
                "Variables en advertencia": int(
                    (
                        drift["Estado"]
                        == "Advertencia"
                    ).sum()
                ),
                "Variables críticas": int(
                    (
                        drift["Estado"]
                        == "Crítico"
                    ).sum()
                ),
                "Variables con alerta": int(
                    drift[
                        "Alerta"
                    ].sum()
                ),
                "Variables monitoreadas": int(
                    len(
                        drift
                    )
                ),
                "Recomendación": drift_recommendation(
                    period_status
                ),
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "Periodo"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# CARGA DEL MODELO
# ============================================================

def load_model(
    path: Path = MODEL_PATH,
):
    """Carga el modelo entrenado y persistido."""
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en: {path}"
        )

    return joblib.load(
        path
    )


# ============================================================
# PREPARACIÓN DE DATOS DE MONITOREO
# ============================================================

def _prepare_external_current_data(
    raw_current: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series | None,
]:
    """
    Prepara un lote externo con el mismo esquema del dataset original.

    La variable objetivo puede estar presente o no. Si no existe,
    se agrega temporalmente para reutilizar clean_data().
    """
    raw_current = raw_current.copy()

    target_available = (
        TARGET
        in raw_current.columns
    )

    if not target_available:
        raw_current[
            TARGET
        ] = np.nan

    current_clean = clean_data(
        raw_current
    )

    if target_available:
        (
            X_current,
            y_current,
        ) = split_features_target(
            current_clean
        )

    else:
        X_current = (
            current_clean
            .drop(
                columns=[
                    TARGET
                ],
                errors="ignore",
            )
        )

        y_current = None

    return (
        X_current,
        y_current,
    )


def prepare_monitoring_data(
    raw_current: pd.DataFrame | None = None,
):
    """
    Prepara la población de referencia y la población actual.

    Modo por defecto:
        TRAIN = referencia histórica.
        TEST = muestra actual simulada.

    Si se proporciona raw_current:
        TRAIN continúa como referencia histórica.
        raw_current actúa como nuevo lote de producción.
    """
    df = clean_data(
        load_data()
    )

    X, y = split_features_target(
        df
    )

    (
        X_train,
        X_test,
        _,
        y_test,
    ) = create_train_test_split(
        X,
        y,
    )

    if raw_current is None:
        return (
            X_train,
            X_test,
            y_test,
            "Simulación holdout: TRAIN vs TEST",
        )

    (
        X_current,
        y_current,
    ) = _prepare_external_current_data(
        raw_current
    )

    return (
        X_train,
        X_current,
        y_current,
        "Monitoreo con lote externo",
    )


# ============================================================
# LECTURA DE ARCHIVO EXTERNO
# ============================================================

def read_uploaded_data(
    uploaded_file,
) -> pd.DataFrame:
    """Lee un archivo CSV o Excel cargado desde Streamlit."""
    file_name = (
        uploaded_file.name
        .lower()
    )

    if file_name.endswith(
        ".csv"
    ):
        return pd.read_csv(
            uploaded_file
        )

    if file_name.endswith(
        (".xlsx", ".xls")
    ):
        return pd.read_excel(
            uploaded_file
        )

    raise ValueError(
        "Formato no soportado. Utilizá CSV, XLSX o XLS."
    )


# ============================================================
# VISUALIZACIÓN
# ============================================================

def plot_distribution_comparison(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature: str,
):
    """
    Construye un gráfico de comparación entre
    distribución histórica y actual.
    """
    if feature in NUMERIC_FEATURES:

        ref = pd.to_numeric(
            reference_df[feature],
            errors="coerce",
        ).dropna()

        cur = pd.to_numeric(
            current_df[feature],
            errors="coerce",
        ).dropna()

        fig, ax = plt.subplots(
            figsize=(9, 4.5)
        )

        ax.hist(
            ref,
            bins=20,
            alpha=0.5,
            density=True,
            label="Referencia histórica",
        )

        ax.hist(
            cur,
            bins=20,
            alpha=0.5,
            density=True,
            label="Población actual",
        )

        ax.set_title(
            f"Distribución histórica vs actual - {feature}"
        )

        ax.set_xlabel(
            feature
        )

        ax.set_ylabel(
            "Densidad"
        )

        ax.legend()

        fig.tight_layout()

        return fig

    ref_dist = (
        reference_df[feature]
        .fillna("MISSING")
        .astype(str)
        .value_counts(
            normalize=True
        )
    )

    cur_dist = (
        current_df[feature]
        .fillna("MISSING")
        .astype(str)
        .value_counts(
            normalize=True
        )
    )

    categories = sorted(
        set(ref_dist.index)
        | set(cur_dist.index)
    )

    comparison = pd.DataFrame(
        {
            "Referencia histórica": (
                ref_dist
                .reindex(
                    categories,
                    fill_value=0,
                )
            ),
            "Población actual": (
                cur_dist
                .reindex(
                    categories,
                    fill_value=0,
                )
            ),
        }
    )

    fig, ax = plt.subplots(
        figsize=(9, 4.5)
    )

    comparison.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title(
        f"Distribución histórica vs actual - {feature}"
    )

    ax.set_xlabel(
        feature
    )

    ax.set_ylabel(
        "Proporción"
    )

    ax.tick_params(
        axis="x",
        rotation=45,
    )

    fig.tight_layout()

    return fig


def render_status_message(
    status: str,
    text: str,
):
    """Renderiza un mensaje visual según el nivel de drift."""
    if status == "Crítico":
        st.error(
            f"🔴 {text}"
        )

    elif status == "Advertencia":
        st.warning(
            f"🟡 {text}"
        )

    else:
        st.success(
            f"🟢 {text}"
        )


# ============================================================
# APLICACIÓN STREAMLIT
# ============================================================

def run_streamlit_app():

    st.set_page_config(
        page_title="Monitoreo de Data Drift",
        page_icon="📊",
        layout="wide",
    )

    st.title(
        "📊 Monitoreo de Data Drift"
    )

    st.caption(
        "Proyecto MLOps - Henry | Avance 3"
    )

    st.info(
        "Por defecto se utiliza TRAIN como población histórica "
        "y TEST como población actual simulada. También podés cargar "
        "un lote externo para simular datos posteriores al despliegue."
    )

    # --------------------------------------------------------
    # Sidebar
    # --------------------------------------------------------

    st.sidebar.header(
        "Configuración"
    )

    uploaded_file = st.sidebar.file_uploader(
        "Cargar población actual (opcional)",
        type=[
            "csv",
            "xlsx",
            "xls",
        ],
        help=(
            "Si no se carga un archivo, la aplicación usa "
            "el conjunto TEST como población actual simulada."
        ),
    )

    # --------------------------------------------------------
    # Preparación de datos
    # --------------------------------------------------------

    try:
        if uploaded_file is None:
            (
                X_reference,
                X_current,
                y_current,
                monitoring_mode,
            ) = prepare_monitoring_data()

        else:
            raw_current = read_uploaded_data(
                uploaded_file
            )

            (
                X_reference,
                X_current,
                y_current,
                monitoring_mode,
            ) = prepare_monitoring_data(
                raw_current=raw_current
            )

    except Exception as error:
        st.error(
            "No fue posible preparar los datos de monitoreo."
        )

        st.exception(
            error
        )

        st.stop()

    # --------------------------------------------------------
    # Modelo
    # --------------------------------------------------------

    try:
        model = load_model()

    except FileNotFoundError:
        st.error(
            "No se encontró `best_model.joblib`."
        )

        st.code(
            "python src/model_train_evaluation.py"
        )

        st.info(
            "Ejecutá primero el entrenamiento para generar "
            "el modelo localmente y luego volvé a iniciar Streamlit."
        )

        st.stop()

    # --------------------------------------------------------
    # Cálculos
    # --------------------------------------------------------

    drift_table = calculate_drift_table(
        reference_df=X_reference,
        current_df=X_current,
    )

    prediction_table = build_prediction_table(
        model=model,
        X_current=X_current,
        y_current=y_current,
    )

    temporal_drift = calculate_temporal_drift(
        reference_df=X_reference,
        current_df=X_current,
    )

    # --------------------------------------------------------
    # Resumen
    # --------------------------------------------------------

    st.subheader(
        "Resumen del monitoreo"
    )

    metric_1, metric_2, metric_3, metric_4 = st.columns(
        4
    )

    max_psi = (
        float(
            drift_table["PSI"].max()
        )
        if not drift_table.empty
        else np.nan
    )

    critical_count = int(
        (
            drift_table["Estado"]
            == "Crítico"
        ).sum()
    )

    warning_count = int(
        (
            drift_table["Estado"]
            == "Advertencia"
        ).sum()
    )

    metric_1.metric(
        "Registros actuales",
        f"{len(X_current):,}".replace(
            ",",
            ".",
        ),
    )

    metric_2.metric(
        "Variables monitoreadas",
        len(
            drift_table
        ),
    )

    metric_3.metric(
        "PSI máximo global",
        (
            f"{max_psi:.4f}"
            if pd.notna(
                max_psi
            )
            else "N/D"
        ),
    )

    metric_4.metric(
        "Variables con alerta",
        critical_count
        + warning_count,
    )

    st.write(
        f"**Modo:** {monitoring_mode}  \n"
        f"**Periodicidad definida:** {MONITORING_FREQUENCY}  \n"
        f"**Mínimo de registros por período:** {MIN_TEMPORAL_SAMPLES}"
    )

    overall_status = classify_drift(
        max_psi
    )

    render_status_message(
        overall_status,
        (
            f"Estado global: {overall_status}. "
            f"PSI máximo = {max_psi:.4f}."
        ),
    )

    # --------------------------------------------------------
    # Tabla de métricas
    # --------------------------------------------------------

    st.subheader(
        "Métricas de Data Drift por variable"
    )

    st.caption(
        "Variables numéricas: KS, PSI y Jensen-Shannon. "
        "Variables categóricas: Chi-cuadrado, PSI y Jensen-Shannon."
    )

    display_columns = [
        "Variable",
        "Tipo",
        "PSI",
        "Jensen-Shannon",
        "KS statistic",
        "KS p-value",
        "Chi2 statistic",
        "Chi2 p-value",
        "Estado",
        "Recomendación",
    ]

    existing_columns = [
        column
        for column in display_columns
        if column in drift_table.columns
    ]

    st.dataframe(
        drift_table[
            existing_columns
        ].round(4),
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # Comparación de distribuciones
    # --------------------------------------------------------

    st.subheader(
        "Distribución histórica vs actual"
    )

    available_features = [
        feature
        for feature in (
            NUMERIC_FEATURES
            + CATEGORICAL_FEATURES
            + ORDINAL_FEATURES
        )
        if (
            feature
            in X_reference.columns
            and feature
            in X_current.columns
        )
    ]

    selected_feature = st.selectbox(
        "Seleccioná una variable",
        options=available_features,
    )

    if selected_feature:
        selected_row = (
            drift_table
            .loc[
                drift_table[
                    "Variable"
                ]
                == selected_feature
            ]
            .iloc[0]
        )

        c1, c2, c3 = st.columns(
            3
        )

        c1.metric(
            "PSI",
            f"{selected_row['PSI']:.4f}",
        )

        c2.metric(
            "Jensen-Shannon",
            f"{selected_row['Jensen-Shannon']:.4f}",
        )

        c3.metric(
            "Estado",
            selected_row[
                "Estado"
            ],
        )

        fig = plot_distribution_comparison(
            reference_df=X_reference,
            current_df=X_current,
            feature=selected_feature,
        )

        st.pyplot(
            fig,
            use_container_width=True,
        )

        plt.close(
            fig
        )

        render_status_message(
            selected_row[
                "Estado"
            ],
            selected_row[
                "Recomendación"
            ],
        )

    # --------------------------------------------------------
    # Evolución temporal
    # --------------------------------------------------------

    st.subheader(
        "Evolución temporal del drift"
    )

    if temporal_drift.empty:
        st.info(
            "No hay suficientes observaciones para construir "
            "el análisis temporal."
        )

    else:
        chart_data = (
            temporal_drift[
                [
                    "Periodo",
                    "PSI promedio",
                    "PSI máximo",
                ]
            ]
            .set_index(
                "Periodo"
            )
        )

        st.line_chart(
            chart_data
        )

        st.dataframe(
            temporal_drift.round(4),
            use_container_width=True,
            hide_index=True,
        )

        temporal_alerts = temporal_drift.loc[
            temporal_drift[
                "Estado período"
            ].isin(
                [
                    "Advertencia",
                    "Crítico",
                ]
            )
        ]

        if temporal_alerts.empty:
            st.success(
                "No se detectaron períodos con drift relevante."
            )

        else:
            st.warning(
                "Se detectaron períodos con desviaciones "
                "que requieren seguimiento."
            )

            for _, row in temporal_alerts.iterrows():
                period_text = (
                    pd.Timestamp(
                        row["Periodo"]
                    )
                    .strftime(
                        "%Y-%m"
                    )
                )

                render_status_message(
                    row[
                        "Estado período"
                    ],
                    (
                        f"{period_text} | "
                        f"PSI máximo = {row['PSI máximo']:.4f} | "
                        f"Variable más afectada: "
                        f"{row['Variable más afectada']} | "
                        f"Variables con alerta: "
                        f"{int(row['Variables con alerta'])}"
                    ),
                )

    # --------------------------------------------------------
    # Pronósticos
    # --------------------------------------------------------

    st.subheader(
        "Datos actuales y pronósticos del modelo"
    )

    st.dataframe(
        prediction_table,
        use_container_width=True,
        hide_index=True,
    )

    if (
        "probabilidad_no_pago"
        in prediction_table.columns
    ):
        st.subheader(
            "Distribución del riesgo pronosticado"
        )

        fig, ax = plt.subplots(
            figsize=(9, 4)
        )

        ax.hist(
            prediction_table[
                "probabilidad_no_pago"
            ].dropna(),
            bins=20,
        )

        ax.set_title(
            "Probabilidad estimada de no pagar a tiempo"
        )

        ax.set_xlabel(
            "Probabilidad"
        )

        ax.set_ylabel(
            "Cantidad de observaciones"
        )

        fig.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True,
        )

        plt.close(
            fig
        )

    # --------------------------------------------------------
    # Metodología
    # --------------------------------------------------------

    with st.expander(
        "Metodología y criterios"
    ):
        st.markdown(
            """
            **PSI**
            - Menor a 0.10: estable.
            - Entre 0.10 y 0.25: advertencia.
            - Mayor o igual a 0.25: crítico.

            **Pruebas complementarias**
            - Kolmogorov-Smirnov para variables numéricas.
            - Chi-cuadrado para variables categóricas.
            - Jensen-Shannon para comparar distribuciones.

            **Importante:** cuando no se carga un segundo dataset,
            TRAIN se utiliza como referencia histórica y TEST como
            población actual simulada. Esto permite validar el proceso
            de monitoreo, pero no representa drift real posterior al
            despliegue.
            """
        )


if __name__ == "__main__":
    run_streamlit_app()
