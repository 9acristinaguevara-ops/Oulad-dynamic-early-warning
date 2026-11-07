
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

modelos = {
    "RF_profundidad_8": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clasificador", RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            class_weight="balanced"
        ))
    ]),
    "Regresion_logistica": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("escalador", StandardScaler()),
        ("clasificador", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])
}

umbrales = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
resultados = []

for nombre, modelo in modelos.items():

    modelo.fit(X_train, y_train)
    probabilidades = modelo.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val, probabilidades)

    for umbral in umbrales:

        predicciones = (probabilidades >= umbral).astype(int)

        resultados.append({
            "modelo": nombre,
            "umbral": umbral,
            "estudiantes": len(y_val),
            "retiradas_reales": int(y_val.sum()),
            "alertas": int(predicciones.sum()),
            "AUC": auc,
            "F1": f1_score(
                y_val, predicciones, zero_division=0
            ),
            "precision": precision_score(
                y_val, predicciones, zero_division=0
            ),
            "recall": recall_score(
                y_val, predicciones, zero_division=0
            )
        })

resumen = pd.DataFrame(resultados)

print("\nEVALUACIÓN DE UMBRALES EN VALIDACIÓN")
print(resumen.round(4).to_string(index=False))

resumen.to_csv(
    "outputs/tables/100_evaluacion_umbrales.csv",
    index=False
)

print("\nArchivo guardado:")
print("outputs/tables/100_evaluacion_umbrales.csv")
