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

# 3. SEPARAR LOS PERIODOS

train = df[df["code_presentation"] == "2013B"].copy()
validacion = df[df["code_presentation"] == "2013J"].copy()
prueba_b = df[df["code_presentation"] == "2014B"].copy()
prueba_j = df[df["code_presentation"] == "2014J"].copy()

# Comprobar que existan datos en todos los periodos

if any(
    periodo.empty
    for periodo in [train, validacion, prueba_b, prueba_j]
):
    raise ValueError("Uno de los periodos no tiene datos.")

# 4. PREPARAR LOS DATOS

X_train = train[variables].fillna(0)
y_train = train[objetivo]

X_validacion = validacion[variables].fillna(0)
y_validacion = validacion[objetivo]

X_prueba_b = prueba_b[variables].fillna(0)
y_prueba_b = prueba_b[objetivo]

X_prueba_j = prueba_j[variables].fillna(0)
y_prueba_j = prueba_j[objetivo]

# 5. ENTRENAR EL MODELO ORIGINAL

modelo = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

modelo.fit(X_train, y_train)

# Obtener probabilidades

prob_validacion = modelo.predict_proba(X_validacion)[:, 1]
prob_b_original = modelo.predict_proba(X_prueba_b)[:, 1]
prob_j_original = modelo.predict_proba(X_prueba_j)[:, 1]

# 6. CALIBRAR LAS PROBABILIDADES

calibrador = LogisticRegression(max_iter=1000)

calibrador.fit(
    prob_validacion.reshape(-1, 1),
    y_validacion
)

prob_validacion_cal = calibrador.predict_proba(
    prob_validacion.reshape(-1, 1)
)[:, 1]

prob_b_cal = calibrador.predict_proba(
    prob_b_original.reshape(-1, 1)
)[:, 1]

prob_j_cal = calibrador.predict_proba(
    prob_j_original.reshape(-1, 1)
)[:, 1]

# 7. SELECCIONAR EL UMBRAL EN VALIDACION

umbrales = [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30]

mejor_umbral = umbrales[0]
mejor_f1 = -1

for umbral in umbrales:

    predicciones = (
        prob_validacion_cal >= umbral
    ).astype(int)

    f1 = f1_score(
        y_validacion,
        predicciones,
        zero_division=0
    )

    if f1 > mejor_f1:
        mejor_f1 = f1
        mejor_umbral = umbral

print("\nSELECCION DEL UMBRAL")
print("Umbral seleccionado:", mejor_umbral)
print("F1 en validacion:", round(mejor_f1, 4))

# 8. FUNCION PARA EVALUAR EL MODELO

def evaluar(nombre, periodo, y_real, probabilidades, umbral):

    predicciones = (
        probabilidades >= umbral
    ).astype(int)

    resultados = {
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

    return resultados

# 9. EVALUAR LOS PERIODOS DE PRUEBA

resultados = []

periodos_prueba = [
    ("2014B", y_prueba_b, prob_b_original, prob_b_cal),
    ("2014J", y_prueba_j, prob_j_original, prob_j_cal)
]

for periodo, y_real, prob_original, prob_cal in periodos_prueba:

    resultado_original = evaluar(
        "Logistica_original",
        periodo,
        y_real,
        prob_original,
        0.45
    )

    resultado_calibrado = evaluar(
        "Logistica_calibrada",
        periodo,
        y_real,
        prob_cal,
        mejor_umbral
    )

    resultados.append(resultado_original)
    resultados.append(resultado_calibrado)

# 10. GUARDAR LOS RESULTADOS

tabla = pd.DataFrame(resultados)

ruta_resultados = salida / "106_evaluacion_final.csv"

tabla.to_csv(ruta_resultados, index=False)

print("\nRESULTADOS DE LA EVALUACION FINAL")
print(tabla.round(4).to_string(index=False))

print("\nArchivo guardado en:")
print(ruta_resultados)