# 108_comparacion_baselines.py

import pandas as pd
from pathlib import Path

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

train = df[df["code_presentation"] == "2013B"].copy()
validacion = df[df["code_presentation"] == "2013J"].copy()
prueba_b = df[df["code_presentation"] == "2014B"].copy()
prueba_j = df[df["code_presentation"] == "2014J"].copy()

# 3. DIVIDIR 2013J PARA CALIBRACION Y UMBRAL

from sklearn.model_selection import train_test_split

calibracion, seleccion = train_test_split(
    validacion,
    test_size=0.5,
    random_state=42,
    stratify=validacion[objetivo]
)

# 4. PREPARAR DATOS

X_train = train[variables].fillna(0)
y_train = train[objetivo]

X_cal = calibracion[variables].fillna(0)
y_cal = calibracion[objetivo]

X_seleccion = seleccion[variables].fillna(0)
y_seleccion = seleccion[objetivo]

# 5. ENTRENAR MODELO

modelo = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

modelo.fit(X_train, y_train)

prob_cal = modelo.predict_proba(X_cal)[:, 1]

calibrador = LogisticRegression(max_iter=1000)

calibrador.fit(
    prob_cal.reshape(-1, 1),
    y_cal
)

# 6. ELEGIR UMBRAL CON LOS DATOS DE SELECCION

prob_seleccion_original = modelo.predict_proba(X_seleccion)[:, 1]

prob_seleccion_cal = calibrador.predict_proba(
    prob_seleccion_original.reshape(-1, 1)
)[:, 1]

umbrales = [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30]

mejor_umbral = umbrales[0]
mejor_f1 = -1

for umbral in umbrales:

    predicciones = (prob_seleccion_cal >= umbral).astype(int)

    f1 = f1_score(
        y_seleccion,
        predicciones,
        zero_division=0
    )

    if f1 > mejor_f1:
        mejor_f1 = f1
        mejor_umbral = umbral

# 7. FUNCION DE EVALUACION

def evaluar(nombre, periodo, y_real, probabilidades, umbral):

    predicciones = (probabilidades >= umbral).astype(int)

    return {
        "modelo": nombre,
        "periodo": periodo,
        "estudiantes": len(y_real),
        "retiros_reales": int(y_real.sum()),
        "umbral": umbral,
        "AUC": roc_auc_score(y_real, probabilidades),
        "F1": f1_score(y_real, predicciones, zero_division=0),
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

# 8. EVALUAR LOS DOS PERIODOS DE PRUEBA

resultados = []

tasa_entrenamiento = y_train.mean()

for periodo, datos in [
    ("2014B", prueba_b),
    ("2014J", prueba_j)
]:

    X = datos[variables].fillna(0)
    y = datos[objetivo]

    prob_original = modelo.predict_proba(X)[:, 1]

    prob_calibrada = calibrador.predict_proba(
        prob_original.reshape(-1, 1)
    )[:, 1]

    resultados.append(
        evaluar(
            "Logistica_original",
            periodo,
            y,
            prob_original,
            0.45
        )
    )

    resultados.append(
        evaluar(
            "Logistica_calibrada",
            periodo,
            y,
            prob_calibrada,
            mejor_umbral
        )
    )

    # Referencia: misma probabilidad para todos,
    # calculada exclusivamente con los datos de entrenamiento.

    prob_referencia = [tasa_entrenamiento] * len(y)

    resultados.append({
        "modelo": "Referencia_tasa_entrenamiento",
        "periodo": periodo,
        "estudiantes": len(y),
        "retiros_reales": int(y.sum()),
        "umbral": None,
        "AUC": None,
        "F1": None,
        "precision": None,
        "recall": None,
        "Brier": brier_score_loss(y, prob_referencia),
        "alertas": 0,
        "porcentaje_alertas": 0.0
    })

# 9. GUARDAR TABLA

tabla = pd.DataFrame(resultados)

ruta = salida / "108_comparacion_baselines.csv"

tabla.to_csv(ruta, index=False)

print("\nUMBRAL CALIBRADO:", mejor_umbral)
print("F1 EN SELECCION:", round(mejor_f1, 4))

print("\nCOMPARACION DE MODELOS")
print(tabla.round(4).to_string(index=False))

print("\nArchivo guardado en:")
print(ruta)
