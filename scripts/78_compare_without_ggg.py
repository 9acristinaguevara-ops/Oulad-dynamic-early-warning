
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.pipeline import Pipeline


# 1. Cargar los datos
df = pd.read_parquet(
    "data/interim/week4_combined_features_v4.parquet"
)

# 2. Definir las variables
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

# 3. Separar los periodos
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

# Excluir GGG de los datos de evaluación
validacion_sin_ggg = validacion[
    validacion["code_module"] != "GGG"
].copy()

test_sin_ggg = test[
    test["code_module"] != "GGG"
].copy()

umbrales = [0.20, 0.30, 0.40, 0.50]

resultados = []

# 4. Entrenar y comparar los modelos
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

    # Seleccionar el umbral usando validación sin GGG
    prob_validacion = modelo.predict_proba(
        validacion_sin_ggg[variables]
    )[:, 1]

    reales_validacion = (
        validacion_sin_ggg["target_future_withdrawal"]
    )

    mejor_umbral = 0.20
    mejor_f1 = -1

    for umbral in umbrales:

        predicciones = (
            prob_validacion >= umbral
        ).astype(int)

        valor_f1 = f1_score(
            reales_validacion,
            predicciones,
            zero_division=0
        )

        if valor_f1 > mejor_f1:
            mejor_f1 = valor_f1
            mejor_umbral = umbral

    print("\nMODELO:", nombre)
    print("Umbral seleccionado:", mejor_umbral)
    print("F1 de validación sin GGG:", round(mejor_f1, 3))

    # Reentrenar con 2013B y 2013J
    modelo.fit(
        train_final[variables],
        train_final["target_future_withdrawal"]
    )

    # Evaluar por separado 2014B y 2014J
    for periodo in ["2014B", "2014J"]:

        grupo = test_sin_ggg[
            test_sin_ggg["code_presentation"] == periodo
        ].copy()

        prob = modelo.predict_proba(
            grupo[variables]
        )[:, 1]

        pred = (prob >= mejor_umbral).astype(int)
        reales = grupo["target_future_withdrawal"]

        resultados.append({
            "modelo": nombre,
            "periodo": periodo,
            "umbral": mejor_umbral,
            "estudiantes": len(grupo),
            "retiradas": int(reales.sum()),
            "alertas": int(pred.sum()),
            "precision": precision_score(
                reales, pred, zero_division=0
            ),
            "recall": recall_score(
                reales, pred, zero_division=0
            ),
            "f1": f1_score(
                reales, pred, zero_division=0
            ),
            "auc": roc_auc_score(reales, prob)
        })


# 5. Mostrar resultados
tabla = pd.DataFrame(resultados)

print("\nRESULTADOS SIN GGG")
print(tabla.round(3).to_string(index=False))

print("\nPROMEDIOS POR MODELO")
resumen = tabla.groupby("modelo").agg(
    f1_promedio=("f1", "mean"),
    auc_promedio=("auc", "mean"),
    precision_promedio=("precision", "mean"),
    recall_promedio=("recall", "mean"),
    alertas_totales=("alertas", "sum"),
    estudiantes=("estudiantes", "sum")
).reset_index()

print(resumen.round(3).to_string(index=False))

tabla.to_csv(
    "outputs/tables/comparison_without_ggg_v1.csv",
    index=False
)

resumen.to_csv(
    "outputs/tables/comparison_without_ggg_summary_v1.csv",
    index=False
)

print("\nArchivos guardados correctamente.")
