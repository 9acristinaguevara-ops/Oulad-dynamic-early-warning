
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

df = pd.read_parquet(
    "data/interim/week4_combined_features_v4.parquet"
)

# Variables de actividad
actividad = [
    "semanas_con_actividad",
    "clics_acumulados_0_4",
    "promedio_clics_0_4",
    "dias_activos_0_4",
    "clics_recientes_3_4",
    "clics_anteriores_0_2",
    "has_week4_activity"
]

# Variables de evaluación sin utilizar notas
entregas = [
    "evaluaciones_programadas_34",
    "evaluaciones_entregadas_34",
    "proporcion_entregadas_34",
    "evaluaciones_aprobadas_34"
]

variables = actividad + entregas

# Separación temporal
entrenamiento = df[
    df["code_presentation"].isin(["2013B", "2013J"])
].copy()

prueba = df[
    df["code_presentation"].isin(["2014B", "2014J"])
].copy()

X_train = entrenamiento[variables]
y_train = entrenamiento["target_future_withdrawal"]

X_test = prueba[variables]
y_test = prueba["target_future_withdrawal"]

print("Filas de entrenamiento:", len(X_train))
print("Filas de prueba:", len(X_test))
print("Variables utilizadas:", len(variables))

print("\nDistribución del objetivo en prueba:")
print(y_test.value_counts())

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

    predicciones = modelo.predict(X_test)
    probabilidades = modelo.predict_proba(X_test)[:, 1]

    resultados.append({
        "modelo": nombre,
        "accuracy": accuracy_score(y_test, predicciones),
        "precision": precision_score(
            y_test, predicciones, zero_division=0
        ),
        "recall": recall_score(
            y_test, predicciones, zero_division=0
        ),
        "f1": f1_score(
            y_test, predicciones, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, probabilidades)
    })

    print("\nModelo:", nombre)
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, predicciones))

tabla = pd.DataFrame(resultados)

tabla.to_csv(
    "outputs/tables/temporal_model_comparison_v1.csv",
    index=False
)

print("\nRESULTADOS DE LA EVALUACIÓN TEMPORAL\n")
print(tabla.round(4).to_string(index=False))

print("\nResultados guardados en:")
print("outputs/tables/temporal_model_comparison_v1.csv")
