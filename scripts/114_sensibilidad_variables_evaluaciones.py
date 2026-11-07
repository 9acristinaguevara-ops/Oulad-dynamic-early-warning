
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss
)

# 1. RUTAS

BASE = Path(__file__).resolve().parent.parent

archivo = BASE / "data" / "interim" / "week4_combined_features_v4.parquet"

salida = BASE / "outputs" / "tables"
salida.mkdir(parents=True, exist_ok=True)

# 2. CARGAR DATOS

df = pd.read_parquet(archivo)

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

# 3. CONFIGURACIONES DE VARIABLES

configuraciones = {
    "Modelo_A_nueve_variables": variables,
    "Modelo_B_sin_programadas": [
        v for v in variables
        if v != "evaluaciones_programadas_34"
    ],
    "Modelo_C_sin_evaluaciones": [
        v for v in variables
        if v not in [
            "evaluaciones_programadas_34",
            "evaluaciones_entregadas_34"
        ]
    ]
}

# 4. SEPARAR LOS PERIODOS

train = df[df["code_presentation"] == "2013B"].copy()
validacion = df[df["code_presentation"] == "2013J"].copy()
prueba_b = df[df["code_presentation"] == "2014B"].copy()
prueba_j = df[df["code_presentation"] == "2014J"].copy()

if any(
    periodo.empty
    for periodo in [train, validacion, prueba_b, prueba_j]
):
    raise ValueError("Uno de los periodos no tiene datos.")

# 5. DIVIDIR 2013J EN CALIBRACIÓN Y SELECCIÓN

calibracion, seleccion_umbral = train_test_split(
    validacion,
    test_size=0.5,
    random_state=42,
    stratify=validacion[objetivo]
)

print("Estudiantes para calibración:", len(calibracion))
print("Estudiantes para elegir umbral:", len(seleccion_umbral))

# 6. FUNCIÓN DE EVALUACIÓN

def evaluar(nombre, periodo, y_real, probabilidades, umbral):

    predicciones = (probabilidades >= umbral).astype(int)

    return {
        "modelo": nombre,
        "periodo": periodo,
        "estudiantes": len(y_real),
        "retiros_reales": int(y_real.sum()),
        "umbral": umbral,
        "AUC": roc_auc_score(y_real, probabilidades),
        "F1": f1_score(
            y_real, predicciones, zero_division=0
        ),
        "precision": precision_score(
            y_real, predicciones, zero_division=0
        ),
        "recall": recall_score(
            y_real, predicciones, zero_division=0
        ),
        "Brier": brier_score_loss(y_real, probabilidades),
        "alertas": int(predicciones.sum()),
        "porcentaje_alertas": round(
            predicciones.mean() * 100, 2
        )
    }

# 7. ENTRENAR Y EVALUAR CADA CONFIGURACIÓN

resultados = []

umbrales = [
    0.10, 0.12, 0.15, 0.18,
    0.20, 0.22, 0.25, 0.30
]

for nombre, variables_modelo in configuraciones.items():

    print("\n" + "=" * 60)
    print(nombre)
    print("Número de variables:", len(variables_modelo))

    X_train = train[variables_modelo].fillna(0)
    y_train = train[objetivo]

    X_cal = calibracion[variables_modelo].fillna(0)
    y_cal = calibracion[objetivo]

    X_umbral = seleccion_umbral[variables_modelo].fillna(0)
    y_umbral = seleccion_umbral[objetivo]

    # Entrenar modelo base

    modelo = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    modelo.fit(X_train, y_train)

    # Calibrar con la primera mitad de 2013J

    prob_cal = modelo.predict_proba(X_cal)[:, 1]

    calibrador = LogisticRegression(max_iter=1000)

    calibrador.fit(
        prob_cal.reshape(-1, 1),
        y_cal
    )

    # Seleccionar umbral con la segunda mitad de 2013J

    prob_umbral_original = modelo.predict_proba(X_umbral)[:, 1]

    prob_umbral_cal = calibrador.predict_proba(
        prob_umbral_original.reshape(-1, 1)
    )[:, 1]

    mejor_umbral = umbrales[0]
    mejor_f1 = -1

    for umbral in umbrales:

        predicciones = (
            prob_umbral_cal >= umbral
        ).astype(int)

        f1 = f1_score(
            y_umbral,
            predicciones,
            zero_division=0
        )

        if f1 > mejor_f1:
            mejor_f1 = f1
            mejor_umbral = umbral

    print("Umbral elegido:", mejor_umbral)
    print("F1 en selección:", round(mejor_f1, 4))

    # Evaluar en los periodos independientes

    for periodo, datos in [
        ("2014B", prueba_b),
        ("2014J", prueba_j)
    ]:

        X_test = datos[variables_modelo].fillna(0)
        y_test = datos[objetivo]

        probabilidades_originales = modelo.predict_proba(
            X_test
        )[:, 1]

        probabilidades_calibradas = calibrador.predict_proba(
            probabilidades_originales.reshape(-1, 1)
        )[:, 1]

        resultados.append(
            evaluar(
                nombre,
                periodo,
                y_test,
                probabilidades_calibradas,
                mejor_umbral
            )
        )

# 8. GUARDAR RESULTADOS

tabla = pd.DataFrame(resultados)

ruta = salida / "114_sensibilidad_variables_evaluaciones.csv"

tabla.to_csv(ruta, index=False)

print("\nRESULTADOS FINALES")
print(tabla.round(4).to_string(index=False))

print("\nArchivo guardado en:")
print(ruta)
