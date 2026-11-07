
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
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

probabilidades = modelo.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, probabilidades)

resultados = []

for umbral in [
    0.40, 0.45, 0.50, 0.55, 0.60,
    0.65, 0.70, 0.75, 0.80, 0.85, 0.90
]:
    predicciones = (probabilidades >= umbral).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, predicciones, labels=[0, 1]
    ).ravel()

    alertas = int(predicciones.sum())

    resultados.append({
        "umbral": umbral,
        "AUC": auc,
        "alertas": alertas,
        "porcentaje_alertado": alertas / len(y_val) * 100,
        "verdaderos_positivos": int(tp),
        "falsas_alarmas": int(fp),
        "retiradas_no_detectadas": int(fn),
        "precision": precision_score(
            y_val, predicciones, zero_division=0
        ),
        "recall": recall_score(
            y_val, predicciones, zero_division=0
        ),
        "F1": f1_score(
            y_val, predicciones, zero_division=0
        )
    })

resumen = pd.DataFrame(resultados)

print("\nEVALUACIÓN OPERATIVA DE UMBRALES")
print(resumen.round(4).to_string(index=False))

resumen.to_csv(
    "outputs/tables/101_umbral_operativo.csv",
    index=False
)

print("\nArchivo guardado:")
print("outputs/tables/101_umbral_operativo.csv")
