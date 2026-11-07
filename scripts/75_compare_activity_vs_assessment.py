
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.pipeline import Pipeline


# 1. Cargar los datos
df = pd.read_parquet(
    "data/interim/week4_combined_features_v4.parquet"
)


# 2. Separar los periodos
train_inicial = df[
    df["code_presentation"] == "2013B"
].copy()

validacion = df[
    df["code_presentation"] == "2013J"
].copy()

train_final = df[
    df["code_presentation"].isin(["2013B", "2013J"])
].copy()

test = df[
    df["code_presentation"].isin(["2014B", "2014J"])
].copy()


# 3. Definir las variables de cada modelo
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

umbrales_posibles = [0.20, 0.30, 0.40, 0.50]


# 4. Seleccionar un umbral por modelo y modulo
umbrales_elegidos = {}

for nombre, variables in modelos.items():

    modelo = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clasificador", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    modelo.fit(
        train_inicial[variables],
        train_inicial["target_future_withdrawal"]
    )

    probabilidades = modelo.predict_proba(
        validacion[variables]
    )[:, 1]

    validacion_resultados = validacion[
        ["code_module", "target_future_withdrawal"]
    ].copy()

    validacion_resultados["probabilidad"] = probabilidades

    # Umbral global de respaldo, elegido con toda la validacion
    mejor_global = 0.20
    mejor_f1_global = -1

    for umbral in umbrales_posibles:

        predicciones = (
            validacion_resultados["probabilidad"] >= umbral
        ).astype(int)

        valor_f1 = f1_score(
            validacion_resultados["target_future_withdrawal"],
            predicciones,
            zero_division=0
        )

        if valor_f1 > mejor_f1_global:
            mejor_f1_global = valor_f1
            mejor_global = umbral

    # Umbrales por modulo
    for modulo in sorted(validacion["code_module"].unique()):

        grupo = validacion_resultados[
            validacion_resultados["code_module"] == modulo
        ]

        reales = grupo["target_future_withdrawal"]

        mejor_modulo = mejor_global
        mejor_f1_modulo = -1

        if reales.nunique() >= 2:

            for umbral in umbrales_posibles:

                predicciones = (
                    grupo["probabilidad"] >= umbral
                ).astype(int)

                valor_f1 = f1_score(
                    reales,
                    predicciones,
                    zero_division=0
                )

                if valor_f1 > mejor_f1_modulo:
                    mejor_f1_modulo = valor_f1
                    mejor_modulo = umbral

        umbrales_elegidos[
            (nombre, modulo)
        ] = mejor_modulo


# 5. Reentrenar y evaluar los dos modelos
resultados = []

for nombre, variables in modelos.items():

    modelo = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clasificador", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    modelo.fit(
        train_final[variables],
        train_final["target_future_withdrawal"]
    )

    for periodo in ["2014B", "2014J"]:

        for modulo in sorted(test["code_module"].unique()):

            grupo = test[
                (test["code_presentation"] == periodo) &
                (test["code_module"] == modulo)
            ].copy()

            if len(grupo) == 0:
                continue

            probabilidades = modelo.predict_proba(
                grupo[variables]
            )[:, 1]

            umbral = umbrales_elegidos.get(
                (nombre, modulo), 0.20
            )

            predicciones = (
                probabilidades >= umbral
            ).astype(int)

            reales = grupo["target_future_withdrawal"]

            precision = precision_score(
                reales, predicciones, zero_division=0
            )

            recall = recall_score(
                reales, predicciones, zero_division=0
            )

            f1 = f1_score(
                reales, predicciones, zero_division=0
            )

            if reales.nunique() == 2:
                auc = roc_auc_score(reales, probabilidades)
            else:
                auc = float("nan")

            resultados.append({
                "modelo": nombre,
                "periodo": periodo,
                "modulo": modulo,
                "umbral": umbral,
                "estudiantes": len(grupo),
                "retiradas": int(reales.sum()),
                "alertas": int(predicciones.sum()),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "auc": auc
            })


# 6. Guardar la comparacion detallada
tabla = pd.DataFrame(resultados)

tabla.to_csv(
    "outputs/tables/activity_vs_assessment_v1.csv",
    index=False
)

print("COMPARACION POR MODELO, PERIODO Y MODULO")
print(tabla.round(3).to_string(index=False))


# 7. Resumen general por modelo
resumen = tabla.groupby("modelo").agg(
    f1_promedio=("f1", "mean"),
    auc_promedio=("auc", "mean"),
    precision_promedio=("precision", "mean"),
    recall_promedio=("recall", "mean"),
    alertas_totales=("alertas", "sum"),
    estudiantes_evaluados=("estudiantes", "sum")
).reset_index()

print("\nRESUMEN GENERAL")
print(resumen.round(3).to_string(index=False))

resumen.to_csv(
    "outputs/tables/activity_vs_assessment_summary_v1.csv",
    index=False
)

print("\nArchivos guardados:")
print("outputs/tables/activity_vs_assessment_v1.csv")
print("outputs/tables/activity_vs_assessment_summary_v1.csv")
