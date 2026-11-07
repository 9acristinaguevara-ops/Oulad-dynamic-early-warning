
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

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

evaluaciones = [
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

for nombre, variables in [
    ("Solo actividad", actividad),
    ("Actividad y evaluaciones", actividad + evaluaciones)
]:
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

    print("\nMODELO:", nombre)

    for modulo in ["CCC", "GGG"]:
        grupo = test[test["code_module"] == modulo]
        real = grupo["target_future_withdrawal"]
        pred = (grupo["probabilidad"] >= 0.5).astype(int)

        print(
            f"{modulo} | "
            f"Precision: {precision_score(real, pred, zero_division=0):.3f} | "
            f"Recall: {recall_score(real, pred, zero_division=0):.3f} | "
            f"F1: {f1_score(real, pred, zero_division=0):.3f} | "
            f"AUC: {roc_auc_score(real, grupo['probabilidad']):.3f}"
        )
