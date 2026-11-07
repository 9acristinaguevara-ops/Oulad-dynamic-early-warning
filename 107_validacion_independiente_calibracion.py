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

# 3. SEPARAR LOS PERIODOS

train = df[df["code_presentation"] == "2013B"].copy()
validacion = df[df["code_presentation"] == "2013J"].copy()
prueba_b = df[df["code_presentation"] == "2014B"].copy()
prueba_j = df[df["code_presentation"] == "2014J"].copy()

if any(
    periodo.empty
    for periodo in [train, validacion, prueba_b, prueba_j]
):
    raise ValueError("Uno de los periodos no tiene datos.")

# 4. DIVIDIR 2013J EN DOS PARTES

calibracion, seleccion_umbral = train_test_split(
    validacion,
    test_size=0.5,
    random_state=42,
    stratify=validacion[objetivo]
)

print("Estudiantes para calibración:", len(calibracion))
print("Estudiantes para elegir umbral:", len(seleccion_umbral))

# 5. PREPARAR LOS DATOS

X_train = train[variables].fillna(0)
y_train = train[objetivo]

X_cal = calibracion[variables].fillna(0)
y_cal = calibracion[objetivo]

X_umbral = seleccion_umbral[variables].fillna(0)
y_umbral = seleccion_umbral[objetivo]

X_b = prueba_b[variables].fillna(0)
y_b = prueba_b[objetivo]

X_j = prueba_j[variables].fillna(0)
y_j = prueba_j[objetivo]

# 6. ENTRENAR EL MODELO BASE

modelo = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

modelo.fit(X_train, y_train)

# 7. CALIBRAR CON LA PRIMERA MITAD DE 2013J

prob_cal = modelo.predict_proba(X_cal)[:, 1]

calibrador = LogisticRegression(max_iter=1000)

calibrador.fit(
    prob_cal.reshape(-1, 1),
    y_cal
)

# 8. SELECCIONAR EL UMBRAL CON LA SEGUNDA MITAD

prob_umbral_original = modelo.predict_proba(X_umbral)[:, 1]

prob_umbral_cal = calibrador.predict_proba(
    prob_umbral_original.reshape(-1, 1)
)[:, 1]

umbrales = [
    0.10, 0.12, 0.15, 0.18,
    0.20, 0.22, 0.25, 0.30
]

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

print("\nSELECCION DEL UMBRAL")
print("Umbral elegido:", mejor_umbral)
print("F1 en selección:", round(mejor_f1, 4))

# 9. FUNCION DE EVALUACION

def evaluar(nombre, periodo, y_real, probabilidades, umbral):

    predicciones = (
        probabilidades >= umbral
    ).astype(int)

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

# 10. EVALUAR 2014B Y 2014J

resultados = []

for periodo, X, y in [
    ("2014B", X_b, y_b),
    ("2014J", X_j, y_j)
]:

    probabilidades_originales = modelo.predict_proba(X)[:, 1]

    probabilidades_calibradas = calibrador.predict_proba(
        probabilidades_originales.reshape(-1, 1)
    )[:, 1]

    resultados.append(
        evaluar(
            "Logistica_original",
            periodo,
            y,
            probabilidades_originales,
            0.45
        )
    )

    resultados.append(
        evaluar(
            "Logistica_calibrada",
            periodo,
            y,
            probabilidades_calibradas,
            mejor_umbral
        )
    )

# 11. GUARDAR RESULTADOS

tabla = pd.DataFrame(resultados)

ruta = salida / "107_validacion_independiente_calibracion.csv"

tabla.to_csv(ruta, index=False)

print("\nRESULTADOS FINALES")
print(tabla.round(4).to_string(index=False))

print("\nArchivo guardado en:")
print(ruta)