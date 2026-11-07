
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

df = pd.read_parquet(
    "data/interim/week4_combined_features_v4.parquet"
)

actividad = [
    "semanas_con_actividad",
    "clics_acumulados_0_4",
    "promedio_clics_0_4",
    "dias_activos_0_4",
    "clics_recientes_3_4",
    "clics_anteriores_0_2",
    "has_week4_activity"
]

entregas = [
    "evaluaciones_programadas_34",
    "evaluaciones_entregadas_34",
    "proporcion_entregadas_34",
    "evaluaciones_aprobadas_34"
]

variables = actividad + entregas

entrenamiento = df[
    df["code_presentation"].isin(["2013B", "2013J"])
]

prueba = df[
    df["code_presentation"].isin(["2014B", "2014J"])
]

X_train = entrenamiento[variables]
y_train = entrenamiento["target_future_withdrawal"]

modelos = {
    "LogisticRegression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("modelo", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "RandomForest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("modelo", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])
}

resultados = []

for nombre, modelo in modelos.items():

    modelo.fit(X_train, y_train)

    for periodo in ["2014B", "2014J"]:

        datos_periodo = prueba[
            prueba["code_presentation"] == periodo
        ]

        X_test = datos_periodo[variables]
        y_test = datos_periodo["target_future_withdrawal"]

        predicciones = modelo.predict(X_test)
        probabilidades = modelo.predict_proba(X_test)[:, 1]

        resultados.append({
            "modelo": nombre,
            "periodo": periodo,
            "estudiantes": len(datos_periodo),
            "retiradas_reales": int(y_test.sum()),
            "precision": precision_score(
                y_test, predicciones, zero_division=0
            ),
            "recall": recall_score(
                y_test, predicciones, zero_division=0
            ),
            "f1": f1_score(
                y_test, predicciones, zero_division=0
            ),
            "roc_auc": roc_auc_score(
                y_test, probabilidades
            )
        })

tabla = pd.DataFrame(resultados)

tabla.to_csv(
    "outputs/tables/temporal_metrics_by_period_v1.csv",
    index=False
)

print("\nRESULTADOS POR PERIODO\n")
print(tabla.round(4).to_string(index=False))

print("\nArchivo guardado en:")
print("outputs/tables/temporal_metrics_by_period_v1.csv")
