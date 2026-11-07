
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, roc_auc_score

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

calibracion = datos[
    datos["code_presentation"] == "2013J"
].copy()

X_train = entrenamiento[variables]
y_train = entrenamiento[objetivo].astype(int)

X_cal = calibracion[variables]
y_cal = calibracion[objetivo].astype(int)

modelo_base = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("escalador", StandardScaler()),
    ("clasificador", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

modelo_base.fit(X_train, y_train)

# Obtener puntuaciones del modelo base
prob_cal = modelo_base.predict_proba(X_cal)[:, 1]

# Aprender la calibración usando solamente 2013J
calibrador = LogisticRegression()
calibrador.fit(prob_cal.reshape(-1, 1), y_cal)

resultados = []

for periodo in ["2013J", "2014B", "2014J"]:

    conjunto = datos[
        datos["code_presentation"] == periodo
    ].copy()

    X = conjunto[variables]
    y = conjunto[objetivo].astype(int)

    prob_original = modelo_base.predict_proba(X)[:, 1]

    prob_calibrada = calibrador.predict_proba(
        prob_original.reshape(-1, 1)
    )[:, 1]

    resultados.append({
        "periodo": periodo,
        "tasa_retirada_real": y.mean(),
        "probabilidad_media_original": prob_original.mean(),
        "probabilidad_media_calibrada": prob_calibrada.mean(),
        "Brier_original": brier_score_loss(y, prob_original),
        "Brier_calibrado": brier_score_loss(y, prob_calibrada),
        "AUC_original": roc_auc_score(y, prob_original),
        "AUC_calibrado": roc_auc_score(y, prob_calibrada)
    })

resumen = pd.DataFrame(resultados)

print("\nCOMPARACIÓN DE CALIBRACIÓN")
print(resumen.round(4).to_string(index=False))

resumen.to_csv(
    "outputs/tables/104_calibracion_logistica.csv",
    index=False
)

print("\nArchivo guardado:")
print("outputs/tables/104_calibracion_logistica.csv")
