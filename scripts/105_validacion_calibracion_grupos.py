
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss

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

prob_cal = modelo.predict_proba(X_cal)[:, 1]

calibrador = LogisticRegression()
calibrador.fit(prob_cal.reshape(-1, 1), y_cal)

resultados = []

for periodo in ["2014B", "2014J"]:

    conjunto = datos[
        datos["code_presentation"] == periodo
    ].copy()

    X = conjunto[variables]
    y = conjunto[objetivo].astype(int)

    prob_original = modelo.predict_proba(X)[:, 1]

    prob_calibrada = calibrador.predict_proba(
        prob_original.reshape(-1, 1)
    )[:, 1]

    evaluacion = pd.DataFrame({
        "probabilidad": prob_calibrada,
        "retirada_real": y.to_numpy()
    })

    evaluacion["grupo"] = pd.qcut(
        evaluacion["probabilidad"],
        q=5,
        labels=False,
        duplicates="drop"
    )

    resumen = evaluacion.groupby("grupo").agg(
        estudiantes=("retirada_real", "size"),
        probabilidad_media=("probabilidad", "mean"),
        tasa_retirada_real=("retirada_real", "mean")
    ).reset_index()

    resumen["periodo"] = periodo
    resultados.append(resumen)

    print("\nPERIODO:", periodo)
    print(resumen.round(4).to_string(index=False))

    brier_modelo = brier_score_loss(y, prob_calibrada)
    brier_base = brier_score_loss(
        y, [y.mean()] * len(y)
    )

    print("Brier del modelo:", round(brier_modelo, 4))
    print("Brier de referencia:", round(brier_base, 4))

tabla = pd.concat(resultados, ignore_index=True)

tabla.to_csv(
    "outputs/tables/105_validacion_calibracion_grupos.csv",
    index=False
)

print("\nArchivo guardado:")
print("outputs/tables/105_validacion_calibracion_grupos.csv")
