
# 109_evaluacion_capacidad_alertas.py

import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

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

# 3. SEPARAR VALIDACION

calibracion, seleccion = train_test_split(
    validacion,
    test_size=0.5,
    random_state=42,
    stratify=validacion[objetivo]
)

# 4. PREPARAR LOS DATOS

X_train = train[variables].fillna(0)
y_train = train[objetivo]

X_cal = calibracion[variables].fillna(0)
y_cal = calibracion[objetivo]

# 5. ENTRENAR MODELO

modelo = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

modelo.fit(X_train, y_train)

# 6. CALIBRAR PROBABILIDADES

prob_cal = modelo.predict_proba(X_cal)[:, 1]

calibrador = LogisticRegression(max_iter=1000)

calibrador.fit(
    prob_cal.reshape(-1, 1),
    y_cal
)

# 7. EVALUAR CAPACIDAD DE ALERTAS

def evaluar_capacidad(periodo, datos):

    X = datos[variables].fillna(0)
    y = datos[objetivo].reset_index(drop=True)

    prob_original = modelo.predict_proba(X)[:, 1]

    prob_calibrada = calibrador.predict_proba(
        prob_original.reshape(-1, 1)
    )[:, 1]

    tabla_estudiantes = pd.DataFrame({
        "retiro_real": y,
        "probabilidad": prob_calibrada
    })

    tabla_estudiantes = tabla_estudiantes.sort_values(
        "probabilidad",
        ascending=False
    ).reset_index(drop=True)

    resultados = []

    for porcentaje in [10, 20, 30]:

        cantidad = round(len(tabla_estudiantes) * porcentaje / 100)

        seleccionados = tabla_estudiantes.head(cantidad)

        alertas = len(seleccionados)
        retiros_detectados = int(seleccionados["retiro_real"].sum())
        retiros_totales = int(y.sum())

        precision = retiros_detectados / alertas if alertas > 0 else 0

        recall = (
            retiros_detectados / retiros_totales
            if retiros_totales > 0 else 0
        )

        resultados.append({
            "periodo": periodo,
            "capacidad_porcentaje": porcentaje,
            "estudiantes_totales": len(tabla_estudiantes),
            "estudiantes_atendidos": alertas,
            "retiros_detectados": retiros_detectados,
            "retiros_totales": retiros_totales,
            "precision": round(precision, 4),
            "recall": round(recall, 4)
        })

    return resultados


# 8. APLICAR A LOS PERIODOS DE PRUEBA

resultados = []

resultados.extend(evaluar_capacidad("2014B", prueba_b))
resultados.extend(evaluar_capacidad("2014J", prueba_j))

tabla = pd.DataFrame(resultados)

# 9. GUARDAR RESULTADOS

ruta = salida / "109_evaluacion_capacidad_alertas.csv"

tabla.to_csv(ruta, index=False)

print("\nEVALUACION SEGUN CAPACIDAD DE ATENCION")
print(tabla.to_string(index=False))

print("\nArchivo guardado en:")
print(ruta)
