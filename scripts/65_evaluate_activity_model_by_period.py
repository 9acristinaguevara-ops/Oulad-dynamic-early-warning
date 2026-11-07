
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

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
    "has_week4_activity"
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

test["prediccion"] = (
    test["probabilidad"] >= 0.5
).astype(int)

print("MODELO SOLO DE ACTIVIDAD: RESULTADOS POR PERIODO\n")

for periodo in ["2014B", "2014J"]:
    for modulo in sorted(test["code_module"].unique()):
        grupo = test[
            (test["code_presentation"] == periodo) &
            (test["code_module"] == modulo)
        ]
        if len(grupo) == 0:
            continue
        real = grupo["target_future_withdrawal"]
        pred = grupo["prediccion"]
        prob = grupo["probabilidad"]

        auc = (
            roc_auc_score(real, prob)
            if real.nunique() == 2
            else None
        )

        print(
            f"Periodo: {periodo} | Modulo: {modulo} | "
            f"Estudiantes: {len(grupo)} | "
            f"Retiradas: {int(real.sum())} | "
            f"Alertas: {int(pred.sum())} | "
            f"Precision: {precision_score(real, pred, zero_division=0):.3f} | "
            f"Recall: {recall_score(real, pred, zero_division=0):.3f} | "
            f"F1: {f1_score(real, pred, zero_division=0):.3f} | "
            f"AUC: {auc if auc is not None else 'No calculable'}"
        )
