
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

df = pd.read_parquet(
    "data/interim/week4_combined_features_v4.parquet"
)

variables = [
    "semanas_con_actividad",
    "clics_acumulados_0_4",
    "promedio_clics_0_4",
    "dias_activos_0_4",
    "clics_recientes_3_4",
    "clics_anteriores_0_2",
    "has_week4_activity",
    "evaluaciones_programadas_34",
    "evaluaciones_entregadas_34",
    "proporcion_entregadas_34",
    "evaluaciones_aprobadas_34"
]

train = df[
    df["code_presentation"].isin(["2013B", "2013J"])
]

test = df[
    df["code_presentation"].isin(["2014B", "2014J"])
].copy()

modelo = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("forest", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

modelo.fit(
    train[variables],
    train["target_future_withdrawal"]
)

test["probabilidad"] = modelo.predict_proba(
    test[variables]
)[:, 1]

print("PROBABILIDADES POR MODULO Y PERIODO\n")

for periodo in ["2014B", "2014J"]:
    for modulo in ["CCC", "GGG"]:
        grupo = test[
            (test["code_presentation"] == periodo) &
            (test["code_module"] == modulo)
        ]

        print(f"\nPeriodo: {periodo} | Modulo: {modulo}")
        print("Estudiantes:", len(grupo))
        print(
            "Retiradas reales:",
            int(grupo["target_future_withdrawal"].sum())
        )
        print(
            "Probabilidad minima:",
            round(grupo["probabilidad"].min(), 3)
        )
        print(
            "Probabilidad media:",
            round(grupo["probabilidad"].mean(), 3)
        )
        print(
            "Probabilidad maxima:",
            round(grupo["probabilidad"].max(), 3)
        )

        print(
            "Alertas con umbral 0.30:",
            int((grupo["probabilidad"] >= 0.30).sum())
        )
        print(
            "Alertas con umbral 0.50:",
            int((grupo["probabilidad"] >= 0.50).sum())
        )
