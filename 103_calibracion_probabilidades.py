
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

resultados = []

for periodo in ["2013J", "2014B", "2014J"]:

    conjunto = datos[
        datos["code_presentation"] == periodo
    ].copy()

    X = conjunto[variables]
    y = conjunto[objetivo].astype(int)

    probabilidades = modelo.predict_proba(X)[:, 1]

    evaluacion = pd.DataFrame({
        "probabilidad": probabilidades,
        "retirada_real": y.to_numpy()
    })

    evaluacion["grupo_riesgo"] = pd.cut(
        evaluacion["probabilidad"],
        bins=[-0.001, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0],
        labels=[
            "0-20%", "20-30%", "30-40%",
            "40-50%", "50-60%", "60-100%"
        ]
    )

    resumen = evaluacion.groupby(
        "grupo_riesgo",
        observed=False
    ).agg(
        estudiantes=("retirada_real", "size"),
        probabilidad_media=("probabilidad", "mean"),
        tasa_retirada_real=("retirada_real", "mean")
    ).reset_index()

    resumen["periodo"] = periodo
    resultados.append(resumen)

    print("\nPERIODO:", periodo)
    print(resumen.round(4).to_string(index=False))

tabla_final = pd.concat(resultados, ignore_index=True)

tabla_final.to_csv(
    "outputs/tables/103_calibracion_probabilidades.csv",
    index=False
)

print("\nArchivo guardado:")
print("outputs/tables/103_calibracion_probabilidades.csv")
