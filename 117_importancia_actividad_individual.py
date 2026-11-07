
# 117_importancia_actividad_individual.py

import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss
)

# --------------------------------------------------
# 1. RUTAS Y CONFIGURACION
# --------------------------------------------------

BASE = Path(__file__).resolve().parents[1]

ARCHIVO_ENTRADA = (
    BASE / "data" / "interim" /
    "week4_combined_features_v4.parquet"
)

CARPETA_SALIDA = BASE / "outputs" / "tables"
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

VARIABLE_OBJETIVO = "target_future_withdrawal"

VARIABLES_MODELO = [
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

VARIABLES_ACTIVIDAD = [
    "week3_clicks",
    "week4_clicks",
    "week4_active_days",
    "week4_activity_records",
    "has_week4_activity",
    "weekly_change",
    "clics_recientes_3_4"
]

UMBRAL = 0.15
REPETICIONES = 30

# --------------------------------------------------
# 2. CARGAR DATOS
# --------------------------------------------------

df = pd.read_parquet(ARCHIVO_ENTRADA)

columnas_necesarias = (
    VARIABLES_MODELO
    + [
        VARIABLE_OBJETIVO,
        "code_module",
        "code_presentation"
    ]
)

for columna in columnas_necesarias:
    if columna not in df.columns:
        raise ValueError(
            f"No se encontró la columna: {columna}"
        )

print("DATOS CARGADOS")
print("Registros:", len(df))
print()

# --------------------------------------------------
# 3. SEPARAR ENTRENAMIENTO Y VALIDACION
# --------------------------------------------------

entrenamiento = df[
    df["code_presentation"] == "2013B"
].copy()

validacion = df[
    df["code_presentation"] == "2013J"
].copy()

X_train = entrenamiento[VARIABLES_MODELO].fillna(0)
y_train = entrenamiento[VARIABLE_OBJETIVO]

X_validacion = validacion[VARIABLES_MODELO].fillna(0)
y_validacion = validacion[VARIABLE_OBJETIVO]

X_calibracion, X_umbral, y_calibracion, y_umbral = (
    train_test_split(
        X_validacion,
        y_validacion,
        test_size=0.5,
        random_state=42,
        stratify=y_validacion
    )
)

# --------------------------------------------------
# 4. ENTRENAR MODELO ORIGINAL
# --------------------------------------------------

modelo = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

modelo.fit(X_train, y_train)

# --------------------------------------------------
# 5. CALIBRAR PROBABILIDADES
# --------------------------------------------------

prob_calibracion = modelo.predict_proba(
    X_calibracion
)[:, 1]

calibrador = LogisticRegression()

calibrador.fit(
    prob_calibracion.reshape(-1, 1),
    y_calibracion
)

# --------------------------------------------------
# 6. PREPARAR CONJUNTOS DE PRUEBA
# --------------------------------------------------

pruebas = {
    "2014B": df[
        df["code_presentation"] == "2014B"
    ].copy(),

    "2014J": df[
        df["code_presentation"] == "2014J"
    ].copy()
}

# --------------------------------------------------
# 7. FUNCION PARA EVALUAR EL MODELO
# --------------------------------------------------

def evaluar(y_real, probabilidades):

    predicciones = (
        probabilidades >= UMBRAL
    ).astype(int)

    return {
        "AUC": roc_auc_score(
            y_real, probabilidades
        ),
        "F1": f1_score(
            y_real, predicciones,
            zero_division=0
        ),
        "Precision": precision_score(
            y_real, predicciones,
            zero_division=0
        ),
        "Recall": recall_score(
            y_real, predicciones,
            zero_division=0
        ),
        "Brier": brier_score_loss(
            y_real, probabilidades
        )
    }

# --------------------------------------------------
# 8. PERMUTACION DENTRO DE CADA GRUPO
# --------------------------------------------------

def permutar_dentro_de_grupos(
    datos,
    variable,
    semilla
):

    datos_permutados = datos.copy()

    generador = np.random.default_rng(semilla)

    grupos = datos_permutados.groupby(
        ["code_module", "code_presentation"]
    ).groups

    for indices in grupos.values():

        valores = datos_permutados.loc[
            indices, variable
        ].to_numpy()

        valores_permutados = generador.permutation(
            valores
        )

        datos_permutados.loc[
            indices, variable
        ] = valores_permutados

    return datos_permutados

# --------------------------------------------------
# 9. EVALUACION DE IMPORTANCIA
# --------------------------------------------------

resultados = []

for periodo, datos_test in pruebas.items():

    print("EVALUANDO PERIODO:", periodo)

    X_test = datos_test[
        VARIABLES_MODELO
    ].fillna(0)

    y_test = datos_test[
        VARIABLE_OBJETIVO
    ]

    # Probabilidades originales
    prob_originales = modelo.predict_proba(
        X_test
    )[:, 1]

    prob_originales = calibrador.predict_proba(
        prob_originales.reshape(-1, 1)
    )[:, 1]

    metricas_originales = evaluar(
        y_test,
        prob_originales
    )

    print("Modelo original:", metricas_originales)

    # Guardar resultado original
    resultados.append({
        "periodo": periodo,
        "variable": "Modelo_original",
        "AUC_original": metricas_originales["AUC"],
        "AUC_permutado": metricas_originales["AUC"],
        "disminucion_AUC": 0,
        "F1_original": metricas_originales["F1"],
        "F1_permutado": metricas_originales["F1"],
        "disminucion_F1": 0,
        "Brier_original": metricas_originales["Brier"],
        "Brier_permutado": metricas_originales["Brier"]
    })

    # Permutar cada variable individual
    for variable in VARIABLES_ACTIVIDAD:

        resultados_repeticiones = []

        for repeticion in range(REPETICIONES):

            datos_permutados = permutar_dentro_de_grupos(
                datos_test,
                variable,
                semilla=42 + repeticion
            )

            X_permutado = datos_permutados[
                VARIABLES_MODELO
            ].fillna(0)

            probabilidades = modelo.predict_proba(
                X_permutado
            )[:, 1]

            probabilidades = calibrador.predict_proba(
                probabilidades.reshape(-1, 1)
            )[:, 1]

            metricas = evaluar(
                y_test,
                probabilidades
            )

            resultados_repeticiones.append(metricas)

        promedio = pd.DataFrame(
            resultados_repeticiones
        ).mean()

        resultados.append({
            "periodo": periodo,
            "variable": variable,
            "AUC_original": metricas_originales["AUC"],
            "AUC_permutado": promedio["AUC"],
            "disminucion_AUC": (
                metricas_originales["AUC"]
                - promedio["AUC"]
            ),
            "F1_original": metricas_originales["F1"],
            "F1_permutado": promedio["F1"],
            "disminucion_F1": (
                metricas_originales["F1"]
                - promedio["F1"]
            ),
            "Brier_original": metricas_originales["Brier"],
            "Brier_permutado": promedio["Brier"]
        })

        print(
            "Variable analizada:",
            variable,
            "| Disminucion AUC:",
            round(
                metricas_originales["AUC"]
                - promedio["AUC"], 4
            )
        )

    print()

# --------------------------------------------------
# 10. GUARDAR RESULTADOS
# --------------------------------------------------

tabla = pd.DataFrame(resultados)

archivo_salida = (
    CARPETA_SALIDA /
    "117_importancia_actividad_individual.csv"
)

tabla.to_csv(
    archivo_salida,
    index=False
)

print("RESULTADOS FINALES")
print(tabla.to_string(index=False))

print()
print("ARCHIVO GUARDADO EN:")
print(archivo_salida)
