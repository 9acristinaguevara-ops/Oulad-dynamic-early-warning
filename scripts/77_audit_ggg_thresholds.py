
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline


# Cargar los datos
df = pd.read_parquet(
    "data/interim/week4_combined_features_v4.parquet"
)

actividad = [
    "week3_clicks",
    "week4_clicks",
    "week4_active_days",
    "week4_activity_records",
    "has_week4_activity",
    "weekly_change",
    "clics_recientes_3_4"
]

participacion = [
    "evaluaciones_programadas_34",
    "evaluaciones_entregadas_34"
]

modelos = {
    "A_solo_actividad": actividad,
    "B_actividad_participacion": actividad + participacion
}

train = df[df["code_presentation"] == "2013B"].copy()
validacion = df[df["code_presentation"] == "2013J"].copy()
train_final = df[
    df["code_presentation"].isin(["2013B", "2013J"])
].copy()

test = df[
    (df["code_module"] == "GGG") &
    (df["code_presentation"].isin(["2014B", "2014J"]))
].copy()

umbrales = [0.20, 0.30, 0.40, 0.50]

for nombre, variables in modelos.items():

    print("\nMODELO:", nombre)

    modelo = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clasificador", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    modelo.fit(
        train[variables],
        train["target_future_withdrawal"]
    )

    # Elegir el umbral usando solamente la validacion
    prob_validacion = modelo.predict_proba(
        validacion[variables]
    )[:, 1]

    mejor_umbral = 0.20
    mejor_f1 = -1

    for umbral in umbrales:
        pred = (prob_validacion >= umbral).astype(int)

        valor_f1 = f1_score(
            validacion["target_future_withdrawal"],
            pred,
            zero_division=0
        )

        print(
            "Umbral:", umbral,
            "| F1 validacion:", round(valor_f1, 3)
        )

        if valor_f1 > mejor_f1:
            mejor_f1 = valor_f1
            mejor_umbral = umbral

    print("Mejor umbral global:", mejor_umbral)

    # Entrenar con ambos periodos de entrenamiento
    modelo.fit(
        train_final[variables],
        train_final["target_future_withdrawal"]
    )

    for periodo in ["2014B", "2014J"]:

        grupo = test[
            test["code_presentation"] == periodo
        ]

        prob = modelo.predict_proba(
            grupo[variables]
        )[:, 1]

        print("\nPeriodo:", periodo)
        print("Estudiantes:", len(grupo))
        print("Retiradas reales:", int(
            grupo["target_future_withdrawal"].sum()
        ))
        print("Probabilidad minima:", round(prob.min(), 3))
        print("Probabilidad media:", round(prob.mean(), 3))
        print("Probabilidad maxima:", round(prob.max(), 3))

        for umbral in umbrales:
            pred = (prob >= umbral).astype(int)

            print(
                "Umbral:", umbral,
                "| Alertas:", int(pred.sum()),
                "| F1:", round(f1_score(
                    grupo["target_future_withdrawal"],
                    pred,
                    zero_division=0
                ), 3)
            )
