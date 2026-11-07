
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score
)

# Cargar datos
datos = pd.read_parquet(
    "data/interim/week4_combined_features_v4.parquet"
)

variables = [
    "week3_clicks",
    "week4_clicks",
    "week4_active_days",
    "week4_activity_records",
    "has_week4_activity",
    "weekly_change",
    "clics_recientes_3_4",
    "evaluaciones_programadas_34",
    "evaluaciones_entregadas_34"
]

objetivo = "target_future_withdrawal"

# Separar cohortes
entrenamiento = datos[
    datos["code_presentation"] == "2013B"
].copy()

validacion = datos[
    datos["code_presentation"] == "2013J"
].copy()

X_train = entrenamiento[variables]
y_train = entrenamiento[objetivo].astype(int)

X_val = validacion[variables]
y_val = validacion[objetivo].astype(int)

# Definir modelos
modelos = {
    "RF_sin_limite": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ),
    "RF_profundidad_8": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    ),
    "RF_profundidad_5": RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42,
        class_weight="balanced"
    ),
    "RF_hoja_10": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
        class_weight="balanced"
    ),
    "Regresion_logistica": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
}

resultados = []
umbral = 0.20

for nombre, clasificador in modelos.items():

    pasos = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if nombre == "Regresion_logistica":
        pasos.append(("escalador", StandardScaler()))

    pasos.append(("clasificador", clasificador))

    modelo = Pipeline(pasos)

    modelo.fit(X_train, y_train)

    for conjunto, X, y in [
        ("Entrenamiento_2013B", X_train, y_train),
        ("Validacion_2013J", X_val, y_val)
    ]:
        probabilidades = modelo.predict_proba(X)[:, 1]
        predicciones = (probabilidades >= umbral).astype(int)

        resultados.append({
            "modelo": nombre,
            "conjunto": conjunto,
            "AUC": roc_auc_score(y, probabilidades),
            "F1": f1_score(y, predicciones, zero_division=0),
            "precision": precision_score(
                y, predicciones, zero_division=0
            ),
            "recall": recall_score(
                y, predicciones, zero_division=0
            )
        })

resumen = pd.DataFrame(resultados)

print("\nCOMPARACIÓN DE COMPLEJIDAD")
print(resumen.round(4).to_string(index=False))

resumen.to_csv(
    "outputs/tables/99_comparacion_complejidad.csv",
    index=False
)

print("\nArchivo guardado:")
print("outputs/tables/99_comparacion_complejidad.csv")
