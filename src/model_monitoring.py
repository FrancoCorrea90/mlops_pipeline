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

KS_WARNING = 0.15
KS_CRITICAL = 0.25

JS_WARNING = 0.15
JS_CRITICAL = 0.30

CHI2_P_THRESHOLD = 0.05
KS_P_THRESHOLD = 0.05

MONITORING_FREQUENCY = "Mensual"
MIN_TEMPORAL_SAMPLES = 50

TEMPORAL_FEATURES = [
    "anio_prestamo",
    "mes_prestamo",
]

STATUS_ORDER = {
    "Estable": 0,
    "Vigilancia": 1,
    "Drift": 2,
    "Sin datos": -1,
}

STATUS_ICON = {
    "Estable": "🟢",
    "Vigilancia": "🟡",
    "Drift": "🔴",
    "Sin datos": "⚪",
}


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
    histórica y reutiliza esos intervalos en la población actual.
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


def format_number(
    value,
    decimals: int = 4,
) -> str:
    """Formatea métricas sin fallar ante NaN."""
    if pd.isna(value):
        return "N/D"

    return f"{value:.{decimals}f}"


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

    return float(js_divergence)


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

    return float(js_divergence)


# ============================================================
# CLASIFICACIÓN COMBINADA
# ============================================================

def classify_numeric_drift(
    psi: float,
    ks_statistic: float,
    ks_pvalue: float,
    js: float,
) -> tuple[str, list[str]]:
    """
    Combina PSI, KS y Jensen-Shannon.

    Drift:
        alguna métrica de magnitud supera umbral crítico.

    Vigilancia:
        señal intermedia o evidencia estadística significativa.

    Estable:
        no se detectan señales relevantes.
    """
    if all(
        pd.isna(value)
        for value in [
            psi,
            ks_statistic,
            js,
        ]
    ):
        return "Sin datos", []

    critical_signals = []
    warning_signals = []

    if pd.notna(psi):
        if psi >= PSI_CRITICAL:
            critical_signals.append(
                f"PSI={psi:.3f} ≥ {PSI_CRITICAL}"
            )
        elif psi >= PSI_WARNING:
            warning_signals.append(
                f"PSI={psi:.3f} ≥ {PSI_WARNING}"
            )

    if pd.notna(ks_statistic):
        if ks_statistic >= KS_CRITICAL:
            critical_signals.append(
                f"KS={ks_statistic:.3f} ≥ {KS_CRITICAL}"
            )
        elif ks_statistic >= KS_WARNING:
            warning_signals.append(
                f"KS={ks_statistic:.3f} ≥ {KS_WARNING}"
            )

    if pd.notna(js):
        if js >= JS_CRITICAL:
            critical_signals.append(
                f"JS={js:.3f} ≥ {JS_CRITICAL}"
            )
        elif js >= JS_WARNING:
            warning_signals.append(
                f"JS={js:.3f} ≥ {JS_WARNING}"
            )

    # Un p-value significativo de KS por sí solo genera vigilancia.
    if (
        pd.notna(ks_pvalue)
        and ks_pvalue < KS_P_THRESHOLD
        and not critical_signals
    ):
        warning_signals.append(
            f"KS p={ks_pvalue:.3f} < {KS_P_THRESHOLD}"
        )

    if critical_signals:
        return (
            "Drift",
            critical_signals
            + warning_signals,
        )

    if warning_signals:
        return (
            "Vigilancia",
            warning_signals,
        )

    return "Estable", []


def classify_categorical_drift(
    psi: float,
    chi2_pvalue: float,
    js: float,
) -> tuple[str, list[str]]:
    """
    Combina PSI, Chi-cuadrado y Jensen-Shannon.

    Chi² p < 0.05 por sí solo genera vigilancia.
    Si además existe una señal de magnitud, refuerza el drift.
    """
    if all(
        pd.isna(value)
        for value in [
            psi,
            chi2_pvalue,
            js,
        ]
    ):
        return "Sin datos", []

    critical_signals = []
    warning_signals = []

    if pd.notna(psi):
        if psi >= PSI_CRITICAL:
            critical_signals.append(
                f"PSI={psi:.3f} ≥ {PSI_CRITICAL}"
            )
        elif psi >= PSI_WARNING:
            warning_signals.append(
                f"PSI={psi:.3f} ≥ {PSI_WARNING}"
            )

    if pd.notna(js):
        if js >= JS_CRITICAL:
            critical_signals.append(
                f"JS={js:.3f} ≥ {JS_CRITICAL}"
            )
        elif js >= JS_WARNING:
            warning_signals.append(
                f"JS={js:.3f} ≥ {JS_WARNING}"
            )

    chi2_significant = (
        pd.notna(chi2_pvalue)
        and chi2_pvalue
        < CHI2_P_THRESHOLD
    )

    if chi2_significant:
        if (
            critical_signals
            or (
                pd.notna(psi)
                and psi >= PSI_WARNING
            )
            or (
                pd.notna(js)
                and js >= JS_WARNING
            )
        ):
            critical_signals.append(
                f"Chi² p={chi2_pvalue:.3f} < {CHI2_P_THRESHOLD}"
            )

        else:
            warning_signals.append(
                f"Chi² p={chi2_pvalue:.3f} < {CHI2_P_THRESHOLD}"
            )

    if critical_signals:
        return (
            "Drift",
            critical_signals
            + warning_signals,
        )

    if warning_signals:
        return (
            "Vigilancia",
            warning_signals,
        )

    return "Estable", []


def drift_recommendation(
    status: str,
    feature_type: str,
) -> str:
    """Genera recomendaciones operativas según el estado."""
    if status == "Drift":
        if feature_type == "Numérica":
            return (
                "Revisar la distribución y calidad de la variable, "
                "comparar con períodos anteriores y validar cambios "
                "en la fuente. Evaluar el impacto sobre el desempeño "
                "del modelo y considerar reentrenamiento si persiste."
            )

        return (
            "Revisar cambios en categorías, frecuencias y calidad "
            "de origen. Validar categorías nuevas o faltantes y "
            "evaluar impacto sobre el modelo. Considerar "
            "reentrenamiento si el cambio persiste."
        )

    if status == "Vigilancia":
        return (
            "Mantener la variable bajo seguimiento. Revisar su "
            "evolución en los próximos períodos y confirmar si "
            "la señal se sostiene antes de tomar acciones."
        )

    if status == "Estable":
        return (
            "Sin acción inmediata. Continuar con el monitoreo periódico."
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
            "Señales": "",
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

    status, signals = classify_numeric_drift(
        psi=psi,
        ks_statistic=float(
            ks_result.statistic
        ),
        ks_pvalue=float(
            ks_result.pvalue
        ),
        js=js,
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
        "Señales": " | ".join(
            signals
        ),
        "Recomendación": drift_recommendation(
            status=status,
            feature_type="Numérica",
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

    status, signals = classify_categorical_drift(
        psi=psi,
        chi2_pvalue=(
            float(chi2_pvalue)
            if pd.notna(
                chi2_pvalue
            )
            else np.nan
        ),
        js=js,
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
        "Señales": " | ".join(
            signals
        ),
        "Recomendación": drift_recommendation(
            status=status,
            feature_type="Categórica",
        ),
    }


def calculate_drift_table(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> pd.DataFrame:
    """Genera la tabla consolidada de métricas de Data Drift."""
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
        drift_df[
            "Semáforo"
        ] = (
            drift_df["Estado"]
            .map(STATUS_ICON)
            .fillna("⚪")
        )

        drift_df[
            "Nivel"
        ] = (
            drift_df["Estado"]
            .map(STATUS_ORDER)
            .fillna(-1)
        )

        drift_df[
            "Alerta"
        ] = (
            drift_df["Estado"]
            .isin(
                [
                    "Vigilancia",
                    "Drift",
                ]
            )
        )

        drift_df = (
            drift_df
            .sort_values(
                by=[
                    "Nivel",
                    "PSI",
                ],
                ascending=[
                    False,
                    False,
                ],
                na_position="last",
            )
            .reset_index(
                drop=True
            )
        )

    return drift_df


def get_overall_status(
    drift_table: pd.DataFrame,
) -> str:
    """Devuelve el peor estado global observado."""
    if drift_table.empty:
        return "Sin datos"

    if (
        drift_table[
            "Estado"
        ]
        == "Drift"
    ).any():
        return "Drift"

    if (
        drift_table[
            "Estado"
        ]
        == "Vigilancia"
    ).any():
        return "Vigilancia"

    return "Estable"


# ============================================================
# PREDICCIONES
# ============================================================

def build_prediction_table(
    model,
    X_current: pd.DataFrame,
    y_current: pd.Series | None = None,
) -> pd.DataFrame:
    """Devuelve datos actuales junto con pronósticos del modelo."""
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
    Calcula evolución mensual del drift usando una referencia fija.
    anio_prestamo y mes_prestamo se usan solo para construir período.
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

    temp[
        "periodo"
    ] = pd.to_datetime(
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
            subset=[
                "periodo"
            ]
        )
        .groupby(
            "periodo"
        )
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
                    + [
                        "periodo"
                    ]
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
            subset=[
                "PSI"
            ]
        )

        psi_average = (
            float(
                valid_psi[
                    "PSI"
                ].mean()
            )
            if not valid_psi.empty
            else np.nan
        )

        psi_max = (
            float(
                valid_psi[
                    "PSI"
                ].max()
            )
            if not valid_psi.empty
            else np.nan
        )

        period_status = get_overall_status(
            drift
        )

        ranked = (
            drift
            .sort_values(
                by=[
                    "Nivel",
                    "PSI",
                    "Jensen-Shannon",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
                na_position="last",
            )
        )

        most_affected = ranked.iloc[
            0
        ]

        rows.append(
            {
                "Periodo": period,
                "Muestras actuales": sample_size,
                "PSI promedio": psi_average,
                "PSI máximo": psi_max,
                "Variable más afectada": (
                    most_affected[
                        "Variable"
                    ]
                ),
                "Estado período": period_status,
                "Semáforo": STATUS_ICON.get(
                    period_status,
                    "⚪",
                ),
                "Variables en vigilancia": int(
                    (
                        drift[
                            "Estado"
                        ]
                        == "Vigilancia"
                    ).sum()
                ),
                "Variables con drift": int(
                    (
                        drift[
                            "Estado"
                        ]
                        == "Drift"
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
                "Señal principal": (
                    most_affected[
                        "Señales"
                    ]
                ),
                "Recomendación": (
                    most_affected[
                        "Recomendación"
                    ]
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


def get_period_drift_detail(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    selected_period,
) -> pd.DataFrame:
    """Devuelve detalle por variable para un período."""
    temp = current_df.copy()

    temp[
        "periodo"
    ] = pd.to_datetime(
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

    selected_period = pd.Timestamp(
        selected_period
    )

    period_df = temp.loc[
        temp[
            "periodo"
        ]
        == selected_period
    ]

    if period_df.empty:
        return pd.DataFrame()

    reference_features = (
        reference_df
        .drop(
            columns=TEMPORAL_FEATURES,
            errors="ignore",
        )
    )

    period_features = (
        period_df
        .drop(
            columns=(
                TEMPORAL_FEATURES
                + [
                    "periodo"
                ]
            ),
            errors="ignore",
        )
    )

    return calculate_drift_table(
        reference_df=reference_features,
        current_df=period_features,
    )


# ============================================================
# CARGA DE MODELO Y DATOS
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


def _prepare_external_current_data(
    raw_current: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series | None,
]:
    """Prepara un lote externo con el mismo esquema esperado."""
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
    Por defecto:
        TRAIN = referencia histórica.
        TEST = población actual simulada.

    Con raw_current:
        TRAIN = referencia.
        raw_current = población actual externa.
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


def read_uploaded_data(
    uploaded_file,
) -> pd.DataFrame:
    """Lee CSV o Excel cargado desde Streamlit."""
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
        (
            ".xlsx",
            ".xls",
        )
    ):
        return pd.read_excel(
            uploaded_file
        )

    raise ValueError(
        "Formato no soportado. Utilizá CSV, XLSX o XLS."
    )


# ============================================================
# VISUALIZACIONES
# ============================================================

def plot_distribution_comparison(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature: str,
):
    """Compara distribución histórica vs actual."""
    if feature in NUMERIC_FEATURES:
        ref = pd.to_numeric(
            reference_df[
                feature
            ],
            errors="coerce",
        ).dropna()

        cur = pd.to_numeric(
            current_df[
                feature
            ],
            errors="coerce",
        ).dropna()

        fig, ax = plt.subplots(
            figsize=(
                9,
                4.5,
            )
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
        reference_df[
            feature
        ]
        .fillna(
            "MISSING"
        )
        .astype(
            str
        )
        .value_counts(
            normalize=True
        )
    )

    cur_dist = (
        current_df[
            feature
        ]
        .fillna(
            "MISSING"
        )
        .astype(
            str
        )
        .value_counts(
            normalize=True
        )
    )

    categories = sorted(
        set(
            ref_dist.index
        )
        | set(
            cur_dist.index
        )
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
        figsize=(
            9,
            4.5,
        )
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


def plot_temporal_evolution(
    temporal_drift: pd.DataFrame,
):
    """Grafica PSI promedio y máximo a lo largo del tiempo."""
    chart = (
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

    fig, ax = plt.subplots(
        figsize=(
            10,
            4.5,
        )
    )

    chart.plot(
        ax=ax,
        marker="o",
    )

    ax.axhline(
        PSI_WARNING,
        linestyle="--",
        linewidth=1,
        label="Umbral PSI vigilancia",
    )

    ax.axhline(
        PSI_CRITICAL,
        linestyle="--",
        linewidth=1,
        label="Umbral PSI crítico",
    )

    ax.set_title(
        "Evolución temporal del Data Drift"
    )

    ax.set_xlabel(
        "Período"
    )

    ax.set_ylabel(
        "PSI"
    )

    ax.legend()

    fig.tight_layout()

    return fig


def plot_temporal_alert_counts(
    temporal_drift: pd.DataFrame,
):
    """Grafica variables en vigilancia y drift por período."""
    chart = (
        temporal_drift[
            [
                "Periodo",
                "Variables en vigilancia",
                "Variables con drift",
            ]
        ]
        .set_index(
            "Periodo"
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            4.5,
        )
    )

    chart.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title(
        "Variables con alerta por período"
    )

    ax.set_xlabel(
        "Período"
    )

    ax.set_ylabel(
        "Cantidad de variables"
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
    """Muestra mensajes visuales según estado."""
    if status == "Drift":
        st.error(
            f"🔴 {text}"
        )

    elif status == "Vigilancia":
        st.warning(
            f"🟡 {text}"
        )

    elif status == "Estable":
        st.success(
            f"🟢 {text}"
        )

    else:
        st.info(
            f"⚪ {text}"
        )


def build_display_table(
    drift_table: pd.DataFrame,
) -> pd.DataFrame:
    """Selecciona columnas para la tabla principal."""
    columns = [
        "Semáforo",
        "Variable",
        "Tipo",
        "PSI",
        "KS statistic",
        "KS p-value",
        "Jensen-Shannon",
        "Chi2 statistic",
        "Chi2 p-value",
        "Estado",
        "Señales",
    ]

    existing = [
        column
        for column in columns
        if column
        in drift_table.columns
    ]

    return (
        drift_table[
            existing
        ]
        .copy()
        .round(4)
    )


def build_alert_table(
    drift_table: pd.DataFrame,
) -> pd.DataFrame:
    """Genera tabla de alertas y recomendaciones."""
    alerts = drift_table.loc[
        drift_table[
            "Estado"
        ].isin(
            [
                "Vigilancia",
                "Drift",
            ]
        )
    ].copy()

    columns = [
        "Semáforo",
        "Variable",
        "Tipo",
        "Estado",
        "Señales",
        "Recomendación",
    ]

    if alerts.empty:
        return pd.DataFrame(
            columns=columns
        )

    return alerts[
        columns
    ]


def register_internal_notification(
    overall_status: str,
    drift_table: pd.DataFrame,
):
    """Registra una notificación interna si cambian las alertas."""
    alerts = drift_table.loc[
        drift_table[
            "Estado"
        ].isin(
            [
                "Vigilancia",
                "Drift",
            ]
        ),
        [
            "Variable",
            "Estado",
        ],
    ]

    fingerprint = tuple(
        sorted(
            (
                row[
                    "Variable"
                ],
                row[
                    "Estado"
                ],
            )
            for _,
            row
            in alerts.iterrows()
        )
    )

    previous = st.session_state.get(
        "alert_fingerprint"
    )

    if (
        fingerprint
        and fingerprint
        != previous
    ):
        drift_count = int(
            (
                drift_table[
                    "Estado"
                ]
                == "Drift"
            ).sum()
        )

        watch_count = int(
            (
                drift_table[
                    "Estado"
                ]
                == "Vigilancia"
            ).sum()
        )

        message = (
            f"Estado global del lote: {overall_status}. "
            f"Drift: {drift_count} variable(s). "
            f"Vigilancia: {watch_count} variable(s)."
        )

        event = {
            "Fecha": pd.Timestamp.now(),
            "Estado": overall_status,
            "Mensaje": message,
        }

        history = st.session_state.get(
            "notification_history",
            [],
        )

        history.insert(
            0,
            event,
        )

        st.session_state[
            "notification_history"
        ] = history[
            :20
        ]

        st.session_state[
            "alert_fingerprint"
        ] = fingerprint

        if overall_status == "Drift":
            st.toast(
                message,
                icon="🚨",
            )

        else:
            st.toast(
                message,
                icon="⚠️",
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
        "Monitoreo de estabilidad de variables y pronósticos del modelo."
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
            "Si no se carga un archivo, la aplicación utiliza "
            "TEST como población actual simulada."
        ),
    )

    st.sidebar.markdown(
        "---"
    )

    st.sidebar.markdown(
        "**Umbrales principales**"
    )

    st.sidebar.caption(
        f"PSI: vigilancia ≥ {PSI_WARNING} | drift ≥ {PSI_CRITICAL}"
    )

    st.sidebar.caption(
        f"KS: vigilancia ≥ {KS_WARNING} | drift ≥ {KS_CRITICAL}"
    )

    st.sidebar.caption(
        f"JS: vigilancia ≥ {JS_WARNING} | drift ≥ {JS_CRITICAL}"
    )

    st.sidebar.caption(
        f"Chi²: p < {CHI2_P_THRESHOLD} = señal estadística"
    )

    # --------------------------------------------------------
    # Datos
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
            "el modelo localmente."
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

    overall_status = get_overall_status(
        drift_table
    )

    register_internal_notification(
        overall_status=overall_status,
        drift_table=drift_table,
    )

    status_icon = STATUS_ICON.get(
        overall_status,
        "⚪",
    )

    st.markdown(
        f"### Estado global del lote: {status_icon} {overall_status}"
    )

    st.caption(
        f"Modo: {monitoring_mode} · "
        f"Periodicidad: {MONITORING_FREQUENCY} · "
        f"Mínimo temporal: {MIN_TEMPORAL_SAMPLES} registros"
    )

    st.markdown(
        "---"
    )

    (
        tab_metrics,
        tab_temporal,
        tab_alerts,
    ) = st.tabs(
        [
            "📊 1. Visualización de métricas",
            "📅 2. Análisis temporal",
            "🚨 3. Recomendaciones y alertas",
        ]
    )

    # ========================================================
    # TAB 1 - MÉTRICAS
    # ========================================================

    with tab_metrics:

        st.subheader(
            "Resumen de métricas"
        )

        st.info(
            "Este tab resume el estado global del lote actual: compara "
            "la población actual completa contra la referencia histórica. "
            "Por eso puede mostrar estabilidad global aunque existan "
            "desvíos puntuales en algunos períodos."
        )

        stable_count = int(
            (
                drift_table[
                    "Estado"
                ]
                == "Estable"
            ).sum()
        )

        watch_count = int(
            (
                drift_table[
                    "Estado"
                ]
                == "Vigilancia"
            ).sum()
        )

        drift_count = int(
            (
                drift_table[
                    "Estado"
                ]
                == "Drift"
            ).sum()
        )

        max_psi = (
            float(
                drift_table[
                    "PSI"
                ].max()
            )
            if not drift_table.empty
            else np.nan
        )

        (
            col_1,
            col_2,
            col_3,
            col_4,
            col_5,
        ) = st.columns(
            5
        )

        col_1.metric(
            "Variables monitoreadas",
            len(
                drift_table
            ),
        )

        col_2.metric(
            "🟢 Estables",
            stable_count,
        )

        col_3.metric(
            "🟡 Vigilancia",
            watch_count,
        )

        col_4.metric(
            "🔴 Drift",
            drift_count,
        )

        col_5.metric(
            "PSI máximo",
            format_number(
                max_psi
            ),
        )

        render_status_message(
            overall_status,
            (
                f"Estado global del lote: {overall_status}. "
                f"Se detectaron {drift_count} variable(s) con drift "
                f"y {watch_count} en vigilancia."
            ),
        )

        st.markdown(
            "#### Tabla de monitoreo"
        )

        (
            filter_type,
            filter_status,
        ) = st.columns(
            2
        )

        type_options = [
            "Todos"
        ] + sorted(
            drift_table[
                "Tipo"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        status_options = [
            "Todos",
            "Drift",
            "Vigilancia",
            "Estable",
        ]

        selected_type = filter_type.selectbox(
            "Filtrar por tipo",
            options=type_options,
        )

        selected_status = filter_status.selectbox(
            "Filtrar por estado",
            options=status_options,
        )

        filtered = drift_table.copy()

        if selected_type != "Todos":
            filtered = filtered.loc[
                filtered[
                    "Tipo"
                ]
                == selected_type
            ]

        if selected_status != "Todos":
            filtered = filtered.loc[
                filtered[
                    "Estado"
                ]
                == selected_status
            ]

        st.dataframe(
            build_display_table(
                filtered
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Umbrales: PSI ≥ 0.25 · KS ≥ 0.25 · "
            "JS ≥ 0.30 = señales críticas. "
            "Chi² p < 0.05 = evidencia estadística. "
            "🟢 estable · 🟡 vigilancia · 🔴 drift"
        )

        st.markdown(
            "#### Distribución histórica vs actual"
        )

        available_features = drift_table[
            "Variable"
        ].tolist()

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
                .iloc[
                    0
                ]
            )

            (
                m1,
                m2,
                m3,
                m4,
            ) = st.columns(
                4
            )

            m1.metric(
                "Estado",
                (
                    f"{selected_row['Semáforo']} "
                    f"{selected_row['Estado']}"
                ),
            )

            m2.metric(
                "PSI",
                format_number(
                    selected_row[
                        "PSI"
                    ]
                ),
            )

            m3.metric(
                "Jensen-Shannon",
                format_number(
                    selected_row[
                        "Jensen-Shannon"
                    ]
                ),
            )

            if (
                selected_row[
                    "Tipo"
                ]
                == "Numérica"
            ):
                m4.metric(
                    "KS",
                    format_number(
                        selected_row[
                            "KS statistic"
                        ]
                    ),
                )

            else:
                m4.metric(
                    "Chi² p-value",
                    format_number(
                        selected_row[
                            "Chi2 p-value"
                        ]
                    ),
                )

            if selected_row[
                "Señales"
            ]:
                st.caption(
                    f"Señales detectadas: {selected_row['Señales']}"
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

        with st.expander(
            "Datos actuales y pronósticos del modelo"
        ):
            st.dataframe(
                prediction_table,
                use_container_width=True,
                hide_index=True,
            )

    # ========================================================
    # TAB 2 - ANÁLISIS TEMPORAL
    # ========================================================

    with tab_temporal:

        st.subheader(
            "Evolución temporal"
        )

        st.info(
            "El análisis temporal evalúa cada período por separado contra "
            "la referencia histórica general. Puede detectar desvíos locales "
            "que no se observan en el agregado global. Estas diferencias "
            "también pueden reflejar estacionalidad o cambios de composición, "
            "por lo que deben interpretarse junto con el tamaño de muestra "
            "y la persistencia de la señal."
        )

        if temporal_drift.empty:
            st.info(
                "No hay suficientes observaciones para construir "
                "el análisis temporal."
            )

        else:
            (
                t1,
                t2,
                t3,
                t4,
            ) = st.columns(
                4
            )

            t1.metric(
                "Períodos analizados",
                len(
                    temporal_drift
                ),
            )

            t2.metric(
                "Períodos con drift",
                int(
                    (
                        temporal_drift[
                            "Estado período"
                        ]
                        == "Drift"
                    ).sum()
                ),
            )

            t3.metric(
                "Períodos en vigilancia",
                int(
                    (
                        temporal_drift[
                            "Estado período"
                        ]
                        == "Vigilancia"
                    ).sum()
                ),
            )

            temporal_max_psi = temporal_drift[
                "PSI máximo"
            ].max()

            t4.metric(
                "PSI máximo temporal",
                format_number(
                    temporal_max_psi
                ),
            )

            fig = plot_temporal_evolution(
                temporal_drift
            )

            st.pyplot(
                fig,
                use_container_width=True,
            )

            plt.close(
                fig
            )

            fig = plot_temporal_alert_counts(
                temporal_drift
            )

            st.pyplot(
                fig,
                use_container_width=True,
            )

            plt.close(
                fig
            )

            st.markdown(
                "#### Resumen por período"
            )

            temporal_display = temporal_drift[
                [
                    "Semáforo",
                    "Periodo",
                    "Muestras actuales",
                    "PSI promedio",
                    "PSI máximo",
                    "Variable más afectada",
                    "Estado período",
                    "Variables en vigilancia",
                    "Variables con drift",
                    "Variables monitoreadas",
                ]
            ].copy()

            temporal_display[
                "Periodo"
            ] = (
                temporal_display[
                    "Periodo"
                ]
                .dt.strftime(
                    "%Y-%m"
                )
            )

            st.dataframe(
                temporal_display.round(
                    4
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown(
                "#### Detalle de un período"
            )

            period_options = temporal_drift[
                "Periodo"
            ].tolist()

            selected_period = st.selectbox(
                "Seleccioná un período",
                options=period_options,
                format_func=lambda value: (
                    pd.Timestamp(
                        value
                    )
                    .strftime(
                        "%Y-%m"
                    )
                ),
            )

            detail = get_period_drift_detail(
                reference_df=X_reference,
                current_df=X_current,
                selected_period=selected_period,
            )

            if not detail.empty:
                st.dataframe(
                    build_display_table(
                        detail
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

    # ========================================================
    # TAB 3 - ALERTAS Y RECOMENDACIONES
    # ========================================================

    with tab_alerts:

        st.subheader(
            "Alertas y recomendaciones"
        )

        st.markdown(
            "#### Alertas globales del lote"
        )

        st.caption(
            "Estas alertas corresponden a la comparación de la población "
            "actual completa contra la referencia histórica."
        )

        alert_table = build_alert_table(
            drift_table
        )

        if alert_table.empty:
            st.success(
                "🟢 No se detectaron variables que requieran "
                "acción o vigilancia."
            )

        else:
            drift_alerts = alert_table.loc[
                alert_table[
                    "Estado"
                ]
                == "Drift"
            ]

            watch_alerts = alert_table.loc[
                alert_table[
                    "Estado"
                ]
                == "Vigilancia"
            ]

            if not drift_alerts.empty:
                st.error(
                    f"🚨 Se detectó drift en {len(drift_alerts)} "
                    "variable(s)."
                )

                for _, row in drift_alerts.iterrows():
                    with st.expander(
                        (
                            f"🔴 {row['Variable']} "
                            f"({row['Tipo']})"
                        ),
                        expanded=True,
                    ):
                        st.write(
                            f"**Señales:** {row['Señales']}"
                        )

                        st.write(
                            f"**Recomendación:** {row['Recomendación']}"
                        )

            if not watch_alerts.empty:
                st.warning(
                    f"⚠️ Hay {len(watch_alerts)} variable(s) "
                    "en vigilancia."
                )

                for _, row in watch_alerts.iterrows():
                    with st.expander(
                        (
                            f"🟡 {row['Variable']} "
                            f"({row['Tipo']})"
                        )
                    ):
                        st.write(
                            f"**Señales:** {row['Señales']}"
                        )

                        st.write(
                            f"**Recomendación:** {row['Recomendación']}"
                        )

            st.markdown(
                "#### Tabla consolidada de alertas globales"
            )

            st.dataframe(
                alert_table,
                use_container_width=True,
                hide_index=True,
            )

            alerts_csv = (
                alert_table
                .to_csv(
                    index=False
                )
                .encode(
                    "utf-8-sig"
                )
            )

            st.download_button(
                label="⬇️ Descargar alertas en CSV",
                data=alerts_csv,
                file_name="alertas_data_drift.csv",
                mime="text/csv",
            )

        st.markdown(
            "#### Alertas temporales"
        )

        st.caption(
            "Estas alertas corresponden a períodos individuales. Un período "
            "puede presentar vigilancia o drift aunque el lote completo sea "
            "globalmente estable."
        )

        if temporal_drift.empty:
            st.info(
                "No hay suficientes observaciones para generar "
                "alertas temporales."
            )

        else:
            temporal_alerts = temporal_drift.loc[
                temporal_drift[
                    "Estado período"
                ].isin(
                    [
                        "Vigilancia",
                        "Drift",
                    ]
                )
            ].copy()

            if temporal_alerts.empty:
                st.success(
                    "🟢 No se detectaron períodos con señales "
                    "de vigilancia o drift."
                )

            else:
                temporal_drift_count = int(
                    (
                        temporal_alerts[
                            "Estado período"
                        ]
                        == "Drift"
                    ).sum()
                )

                temporal_watch_count = int(
                    (
                        temporal_alerts[
                            "Estado período"
                        ]
                        == "Vigilancia"
                    ).sum()
                )

                if temporal_drift_count:
                    st.error(
                        f"🚨 Se detectó drift en {temporal_drift_count} "
                        "período(s)."
                    )

                if temporal_watch_count:
                    st.warning(
                        f"⚠️ Hay {temporal_watch_count} período(s) "
                        "en vigilancia."
                    )

                temporal_alerts_display = temporal_alerts[
                    [
                        "Semáforo",
                        "Periodo",
                        "Muestras actuales",
                        "Estado período",
                        "Variable más afectada",
                        "PSI máximo",
                        "Variables en vigilancia",
                        "Variables con drift",
                        "Señal principal",
                        "Recomendación",
                    ]
                ].copy()

                temporal_alerts_display[
                    "Periodo"
                ] = (
                    temporal_alerts_display[
                        "Periodo"
                    ]
                    .dt.strftime(
                        "%Y-%m"
                    )
                )

                st.dataframe(
                    temporal_alerts_display.round(
                        4
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        st.markdown(
            "#### Notificaciones internas"
        )

        st.caption(
            "Las notificaciones de esta versión se generan dentro "
            "del dashboard. El envío externo por email, Slack u otro "
            "canal puede incorporarse posteriormente."
        )

        history = st.session_state.get(
            "notification_history",
            [],
        )

        if not history:
            st.info(
                "No se generaron notificaciones en esta sesión."
            )

        else:
            history_df = pd.DataFrame(
                history
            )

            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True,
            )

        with st.expander(
            "Criterios utilizados para el semáforo"
        ):
            st.markdown(
                f"""
                **Variables numéricas**
                - PSI: vigilancia desde `{PSI_WARNING}` y drift desde `{PSI_CRITICAL}`.
                - KS statistic: vigilancia desde `{KS_WARNING}` y drift desde `{KS_CRITICAL}`.
                - Jensen-Shannon: vigilancia desde `{JS_WARNING}` y drift desde `{JS_CRITICAL}`.
                - KS p-value `< {KS_P_THRESHOLD}` se utiliza como señal estadística de vigilancia.

                **Variables categóricas**
                - PSI: vigilancia desde `{PSI_WARNING}` y drift desde `{PSI_CRITICAL}`.
                - Jensen-Shannon: vigilancia desde `{JS_WARNING}` y drift desde `{JS_CRITICAL}`.
                - Chi² p-value `< {CHI2_P_THRESHOLD}` indica diferencia estadísticamente significativa.
                - Chi² por sí solo genera vigilancia; para reforzar una clasificación crítica se combina con una señal de magnitud.

                **Interpretación**
                - 🟢 Estable: sin señales relevantes.
                - 🟡 Vigilancia: cambio moderado o evidencia estadística que requiere seguimiento.
                - 🔴 Drift: cambio de magnitud relevante según alguna de las métricas principales.
                """
            )


if __name__ == "__main__":
    run_streamlit_app()
