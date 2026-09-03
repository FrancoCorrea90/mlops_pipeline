import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from xgboost import XGBClassifier

from ft_engineering import (
    load_data,
    clean_data,
    split_features_target,
    create_train_test_split,
    build_preprocessor,
)


RANDOM_STATE = 42


# ==================================================
# FUNCIONES
# ==================================================

def build_model(estimator):
    """
    Construye un pipeline de Machine Learning compuesto por:
    preprocesamiento + modelo.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def summarize_classification(model_name, model, X_test, y_test):
    """
    Calcula las principales métricas de clasificación.

    Debido al fuerte desbalance del target, se presta especial
    atención a la clase 0: clientes que no pagan a tiempo.
    """

    y_pred = model.predict(X_test)

    results = {
        "Modelo": model_name,

        "Accuracy": accuracy_score(
            y_test,
            y_pred,
        ),

        "Balanced Accuracy": balanced_accuracy_score(
            y_test,
            y_pred,
        ),

        "Precision clase 0": precision_score(
            y_test,
            y_pred,
            pos_label=0,
            zero_division=0,
        ),

        "Recall clase 0": recall_score(
            y_test,
            y_pred,
            pos_label=0,
            zero_division=0,
        ),

        "F1-Score clase 0": f1_score(
            y_test,
            y_pred,
            pos_label=0,
            zero_division=0,
        ),
    }

    # Métricas basadas en probabilidades
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X_test)

        classes = model.named_steps["model"].classes_

        class_0_index = list(classes).index(0)

        probability_class_0 = probabilities[:, class_0_index]

        # Para ROC-AUC y PR-AUC convertimos temporalmente
        # la clase 0 en la clase positiva.
        y_test_class_0 = (y_test == 0).astype(int)

        results["ROC-AUC clase 0"] = roc_auc_score(
            y_test_class_0,
            probability_class_0,
        )

        results["PR-AUC clase 0"] = average_precision_score(
            y_test_class_0,
            probability_class_0,
        )

    return results


# ==================================================
# EJECUCIÓN PRINCIPAL
# ==================================================

if __name__ == "__main__":

    # --------------------------------------------------
    # 1. Preparación de datos
    # --------------------------------------------------

    df = load_data()

    df = clean_data(df)

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = create_train_test_split(
        X,
        y,
    )

    print("Train:", X_train.shape)
    print("Test:", X_test.shape)

    print("\nDistribución de clases en entrenamiento:")

    print(
        y_train
        .value_counts(normalize=True)
        .sort_index()
    )


    # --------------------------------------------------
    # 2. Definición de modelos
    # --------------------------------------------------

    models = {

        # Baseline
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),

        # Modelo de ensamble
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        # Modelo boosting
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


    # --------------------------------------------------
    # 3. Entrenamiento y evaluación
    # --------------------------------------------------

    results = []

    # Guardamos los modelos ya entrenados para poder
    # reutilizarlos luego en las matrices de confusión.
    trained_models = {}

    # Pesos balanceados utilizados por XGBoost.
    xgb_sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    for model_name, estimator in models.items():

        print(f"\nEntrenando: {model_name}")

        model = build_model(estimator)

        if model_name == "XGBoost":

            model.fit(
                X_train,
                y_train,
                model__sample_weight=xgb_sample_weights,
            )

        else:

            model.fit(
                X_train,
                y_train,
            )

        # Guardamos el modelo entrenado.
        trained_models[model_name] = model

        # Calculamos métricas.
        metrics = summarize_classification(
            model_name,
            model,
            X_test,
            y_test,
        )

        results.append(metrics)


    # --------------------------------------------------
    # 4. Tabla comparativa
    # --------------------------------------------------

    results_df = pd.DataFrame(results)

    print("\nResultados comparativos:\n")

    print(
        results_df
        .round(4)
        .to_string(index=False)
    )


    # --------------------------------------------------
    # 5. Matrices de confusión
    # --------------------------------------------------

    print("\nMatrices de confusión:")

    for model_name, model in trained_models.items():

        y_pred = model.predict(X_test)

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1],
        )

        print(f"\n{model_name}:")
        print(cm)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[
                "No paga a tiempo (0)",
                "Paga a tiempo (1)",
            ],
        )

        disp.plot()

        plt.title(
            f"Matriz de confusión - {model_name}"
        )

        plt.tight_layout()

        plt.show()


    # --------------------------------------------------
    # 6. Gráfico comparativo de métricas
    # --------------------------------------------------

    metrics_to_plot = [
        "Balanced Accuracy",
        "Precision clase 0",
        "Recall clase 0",
        "F1-Score clase 0",
        "PR-AUC clase 0",
    ]

    comparison_df = (
        results_df[
            ["Modelo"] + metrics_to_plot
        ]
        .set_index("Modelo")
    )

    comparison_df.plot(
        kind="bar",
        figsize=(11, 6),
    )

    plt.title(
        "Comparación de modelos - métricas de clasificación"
    )

    plt.ylabel("Score")

    plt.ylim(0, 1)

    plt.xticks(
        rotation=0
    )

    plt.legend(
        loc="best"
    )

    plt.tight_layout()

    plt.show()

    # --------------------------------------------------
    # 7. Selección del mejor modelo
    # --------------------------------------------------

    best_model_row = results_df.loc[
        results_df["F1-Score clase 0"].idxmax()
    ]

    best_model_name = best_model_row["Modelo"]
    best_f1_score = best_model_row["F1-Score clase 0"]

    print("\nModelo seleccionado:")

    print(
        f"{best_model_name} "
        f"(F1-Score clase 0 = {best_f1_score:.4f})"
    )

    # XGBoost es seleccionado como modelo candidato al presentar
    # el mejor F1-Score sobre la clase minoritaria.
    # Logistic Regression alcanza un Recall superior, pero genera
    # una cantidad considerablemente mayor de falsos positivos.
    # Random Forest obtiene una Accuracy elevada, aunque presenta
    # un Recall muy bajo para la clase 0.