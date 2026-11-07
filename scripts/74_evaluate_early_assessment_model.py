
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
    "data/interim/week4_early_assessment_v1.parquet"
)


# 2. Separar entrenamiento, validacion y prueba
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


# 3. Definir las caracteristicas
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


print("TAMANO DE LOS CONJUNTOS")
print("Entrenamiento inicial:", len(train_inicial))
print("Validacion:", len(validacion))
print("Entrenamiento final:", len(train_final))
print("Prueba:", len(test))


# 4. Crear y entrenar el modelo inicial
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
    train_inicial[objetivo]
)


# 5. Elegir el umbral usando solo validacion
probabilidades = modelo.predict_proba(
    validacion[variables]
)[:, 1]

umbrales = [0.20, 0.30, 0.40, 0.50]

mejor_umbral = 0.20
mejor_f1 = -1

for umbral in umbrales:

    predicciones = (
        probabilidades >= umbral
    ).astype(int)

    resultado_f1 = f1_score(
        validacion[objetivo],
        predicciones,
        zero_division=0
    )

    print(
        "Umbral:", umbral,
        "| F1 validacion:", round(resultado_f1, 3)
    )

    if resultado_f1 > mejor_f1:
        mejor_f1 = resultado_f1
        mejor_umbral = umbral


print("\nMEJOR UMBRAL DE VALIDACION:", mejor_umbral)
print("F1 de validacion:", round(mejor_f1, 3))


# 6. Entrenar de nuevo con 2013B y 2013J
modelo.fit(
    train_final[variables],
    train_final[objetivo]
)


# 7. Evaluar la prueba final
resultados = []

for periodo in ["2014B", "2014J"]:

    grupo = test[
        test["code_presentation"] == periodo
    ].copy()

    probabilidades = modelo.predict_proba(
        grupo[variables]
    )[:, 1]

    predicciones = (
        probabilidades >= mejor_umbral
    ).astype(int)

    reales = grupo[objetivo]

    precision = precision_score(
        reales, predicciones, zero_division=0
    )

    recall = recall_score(
        reales, predicciones, zero_division=0
    )

    f1 = f1_score(
        reales, predicciones, zero_division=0
    )

    auc = roc_auc_score(
        reales, probabilidades
    )

    resultados.append({
        "modelo": "Actividad + participacion",
        "periodo": periodo,
        "umbral": mejor_umbral,
        "estudiantes": len(grupo),
        "retiradas": int(reales.sum()),
        "alertas": int(predicciones.sum()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    })

    print("\nPERIODO:", periodo)
    print("Estudiantes:", len(grupo))
    print("Retiradas reales:", int(reales.sum()))
    print("Alertas generadas:", int(predicciones.sum()))
    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
    print("F1:", round(f1, 3))
    print("AUC:", round(auc, 3))


# 8. Guardar resultados
tabla = pd.DataFrame(resultados)

tabla.to_csv(
    "outputs/tables/early_assessment_results_v1.csv",
    index=False
)

print("\nRESULTADOS GUARDADOS EN:")
print("outputs/tables/early_assessment_results_v1.csv")
