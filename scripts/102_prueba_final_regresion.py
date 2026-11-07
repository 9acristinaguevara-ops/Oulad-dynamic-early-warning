
import pandas as pd
from sklearn.impute import SimpleImputer
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

X_train = entrenamiento[variables]
y_train = entrenamiento[objetivo].astype(int)

modelo = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("escalador", StandardScaler()),
    ("clasificador", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

modelo.fit(X_train, y_train)

umbral = 0.45
resultados = []

for periodo in ["2013J", "2014B", "2014J"]:

    conjunto = datos[
        datos["code_presentation"] == periodo
    ].copy()

    X = conjunto[variables]
    y = conjunto[objetivo].astype(int)

    probabilidades = modelo.predict_proba(X)[:, 1]
    predicciones = (probabilidades >= umbral).astype(int)

    resultados.append({
        "periodo": periodo,
        "estudiantes": len(y),
        "retiradas_reales": int(y.sum()),
        "alertas": int(predicciones.sum()),
        "porcentaje_alertado": predicciones.mean() * 100,
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

print("\nEVALUACIÓN TEMPORAL CON UMBRAL 0.45")
print(resumen.round(4).to_string(index=False))

resumen.to_csv(
    "outputs/tables/102_prueba_final_regresion.csv",
    index=False
)

print("\nArchivo guardado:")
print("outputs/tables/102_prueba_final_regresion.csv")
