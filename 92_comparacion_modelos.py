
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score
)


# 1. Rutas
ENTRADA = Path("data/interim/week4_combined_features_v4.parquet")
SALIDA = Path("outputs/tables")
SALIDA.mkdir(parents=True, exist_ok=True)


# 2. Cargar datos
df = pd.read_parquet(ENTRADA)

print("Dataset:", len(df), "estudiantes")


# 3. Definir variables
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
    "Modelo_0_referencia": [],
    "Modelo_A_actividad": actividad,
    "Modelo_B_actividad_evaluaciones": actividad + participacion
}


# 4. Preparar periodos
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

print("Entrenamiento inicial:", len(train_inicial))
print("Validación:", len(validacion))
print("Entrenamiento final:", len(train_final))
print("Prueba:", len(test))


# 5. Seleccionar umbrales usando solamente 2013J
umbrales_posibles = [0.20, 0.30, 0.40, 0.50]

umbrales = {}
resultados_validacion = {}

for nombre, variables in modelos.items():

    if nombre == "Modelo_0_referencia":
        modelo = DummyClassifier(strategy="most_frequent")
        modelo.fit(
            train_inicial[["target_future_withdrawal"]],
            train_inicial["target_future_withdrawal"]
        )

        # El modelo de referencia predice siempre la clase mayoritaria.
        # No necesita seleccionar un umbral.
        umbrales[nombre] = None

        pred = modelo.predict(
            validacion[["target_future_withdrawal"]]
        )

        resultados_validacion[nombre] = f1_score(
            validacion["target_future_withdrawal"],
            pred,
            zero_division=0
        )

        continue

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

    mejor_umbral = 0.20
    mejor_f1 = -1

    for umbral in umbrales_posibles:
        pred = (probabilidades >= umbral).astype(int)

        valor_f1 = f1_score(
            validacion["target_future_withdrawal"],
            pred,
            zero_division=0
        )

        if valor_f1 > mejor_f1:
            mejor_f1 = valor_f1
            mejor_umbral = umbral

    umbrales[nombre] = mejor_umbral
    resultados_validacion[nombre] = mejor_f1


# 6. Entrenar y evaluar
resultados = []

for nombre, variables in modelos.items():

    for periodo in ["2014B", "2014J"]:

        grupo = test[
            test["code_presentation"] == periodo
        ].copy()

        reales = grupo["target_future_withdrawal"].astype(int)

        if nombre == "Modelo_0_referencia":
            modelo = DummyClassifier(strategy="most_frequent")

            modelo.fit(
                train_final[["target_future_withdrawal"]],
                train_final["target_future_withdrawal"]
            )

            pred = modelo.predict(
                grupo[["target_future_withdrawal"]]
            )

            probabilidades = modelo.predict_proba(
                grupo[["target_future_withdrawal"]]
            )[:, 1]

            umbral = None

        else:
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

            probabilidades = modelo.predict_proba(
                grupo[variables]
            )[:, 1]

            umbral = umbrales[nombre]
            pred = (probabilidades >= umbral).astype(int)

        resultados.append({
            "modelo": nombre,
            "periodo": periodo,
            "umbral": umbral,
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
            "accuracy": accuracy_score(reales, pred),
            "auc": (
                roc_auc_score(reales, probabilidades)
                if reales.nunique() == 2
                else float("nan")
            )
        })


# 7. Mostrar y guardar resultados
tabla = pd.DataFrame(resultados)

print("\nF1 SELECCIONADO EN VALIDACIÓN")
for nombre in modelos:
    print(
        nombre,
        "| F1:",
        round(resultados_validacion[nombre], 3),
        "| Umbral:",
        umbrales[nombre]
    )

print("\nRESULTADOS EN PRUEBA")
print(tabla.round(3).to_string(index=False))

archivo = SALIDA / "92_comparacion_modelos.csv"
tabla.to_csv(archivo, index=False)

print("\nArchivo guardado:")
print(archivo)
