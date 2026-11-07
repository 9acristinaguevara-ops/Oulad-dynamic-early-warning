
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

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

modelo.fit(train[variables], train["target_future_withdrawal"])

test["probabilidad"] = modelo.predict_proba(test[variables])[:, 1]

print("COMPARACION DE UMBRALES POR MODULO\n")

for periodo in ["2014B", "2014J"]:
    for modulo in ["CCC", "GGG"]:
        grupo = test[
            (test["code_presentation"] == periodo) &
            (test["code_module"] == modulo)
        ]

        y_real = grupo["target_future_withdrawal"]

        print(f"\nPeriodo: {periodo} | Modulo: {modulo}")

        for umbral in [0.30, 0.40, 0.50]:
            y_pred = (
                grupo["probabilidad"] >= umbral
            ).astype(int)

            alertas = int(y_pred.sum())
            verdaderos_positivos = int(
                ((y_real == 1) & (y_pred == 1)).sum()
            )

            print(
                f"Umbral: {umbral:.2f} | "
                f"Alertas: {alertas} | "
                f"Retiradas detectadas: {verdaderos_positivos} | "
                f"Precision: {precision_score(y_real, y_pred, zero_division=0):.3f} | "
                f"Recall: {recall_score(y_real, y_pred, zero_division=0):.3f} | "
                f"F1: {f1_score(y_real, y_pred, zero_division=0):.3f}"
            )
