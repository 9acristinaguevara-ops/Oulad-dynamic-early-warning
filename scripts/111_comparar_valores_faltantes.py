
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

BASE = Path(__file__).resolve().parent.parent

archivo = BASE / "data" / "interim" / "week4_combined_features_v4.parquet"
salida = BASE / "outputs" / "tables"
salida.mkdir(parents=True, exist_ok=True)

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

# Separar las mismas cohortes del experimento original
train = df[df["code_presentation"] == "2013B"].copy()
validacion = df[df["code_presentation"] == "2013J"].copy()
prueba_b = df[df["code_presentation"] == "2014B"].copy()
prueba_j = df[df["code_presentation"] == "2014J"].copy()

if any(x.empty for x in [train, validacion, prueba_b, prueba_j]):
    raise ValueError("Una de las cohortes está vacía.")

# Mantener la misma división de validación
calibracion, seleccion_umbral = train_test_split(
    validacion,
    test_size=0.5,
    random_state=42,
    stratify=validacion[objetivo]
)

# Identificar variables con valores faltantes en entrenamiento
variables_con_faltantes = [
    v for v in variables if train[v].isna().any()
]

print("Variables con faltantes en entrenamiento:")
print(variables_con_faltantes)

# Crear una copia con indicadores de valores faltantes
def preparar(datos, usar_indicadores):
    X = datos[variables].copy()

    if usar_indicadores:
        for v in variables_con_faltantes:
            X["faltante_" + v] = X[v].isna().astype(int)

    return X.fillna(0)


def entrenar_y_evaluar(nombre, usar_indicadores):

    X_train = preparar(train, usar_indicadores)
    X_cal = preparar(calibracion, usar_indicadores)
    X_umbral = preparar(seleccion_umbral, usar_indicadores)
    X_b = preparar(prueba_b, usar_indicadores)
    X_j = preparar(prueba_j, usar_indicadores)

    y_train = train[objetivo]
    y_cal = calibracion[objetivo]
    y_umbral = seleccion_umbral[objetivo]

    modelo = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    modelo.fit(X_train, y_train)

    # Calibración independiente para esta variante
    prob_cal = modelo.predict_proba(X_cal)[:, 1]

    calibrador = LogisticRegression(max_iter=1000)
    calibrador.fit(prob_cal.reshape(-1, 1), y_cal)

    # Selección del umbral en la misma partición
    prob_umbral = modelo.predict_proba(X_umbral)[:, 1]
    prob_umbral = calibrador.predict_proba(
        prob_umbral.reshape(-1, 1)
    )[:, 1]

    umbrales = [
        0.10, 0.12, 0.15, 0.18,
        0.20, 0.22, 0.25, 0.30
    ]

    mejor_umbral = umbrales[0]
    mejor_f1 = -1

    for u in umbrales:
        pred = (prob_umbral >= u).astype(int)
        f1 = f1_score(y_umbral, pred, zero_division=0)

        if f1 > mejor_f1:
            mejor_f1 = f1
            mejor_umbral = u

    print("\nVariante:", nombre)
    print("Umbral seleccionado:", mejor_umbral)
    print("F1 de selección:", round(mejor_f1, 4))

    resultados = []

    for periodo, X, y in [
        ("2014B", X_b, prueba_b[objetivo]),
        ("2014J", X_j, prueba_j[objetivo])
    ]:
        prob = modelo.predict_proba(X)[:, 1]
        prob = calibrador.predict_proba(
            prob.reshape(-1, 1)
        )[:, 1]

        pred = (prob >= mejor_umbral).astype(int)

        resultados.append({
            "variante": nombre,
            "periodo": periodo,
            "estudiantes": len(y),
            "retiros_reales": int(y.sum()),
            "umbral": mejor_umbral,
            "AUC": roc_auc_score(y, prob),
            "F1": f1_score(y, pred, zero_division=0),
            "precision": precision_score(
                y, pred, zero_division=0
            ),
            "recall": recall_score(
                y, pred, zero_division=0
            ),
            "Brier": brier_score_loss(y, prob),
            "alertas": int(pred.sum()),
            "porcentaje_alertas": round(
                pred.mean() * 100, 2
            )
        })

    return resultados


# A: método actual, faltantes reemplazados por cero
resultados_a = entrenar_y_evaluar(
    "A_actual_cero",
    usar_indicadores=False
)

# B: mismo método, añadiendo indicadores de faltantes
resultados_b = entrenar_y_evaluar(
    "B_con_indicadores",
    usar_indicadores=True
)

tabla = pd.DataFrame(resultados_a + resultados_b)

ruta = salida / "111_comparacion_valores_faltantes.csv"
tabla.to_csv(ruta, index=False)

print("\nRESULTADOS COMPARATIVOS")
print(tabla.round(4).to_string(index=False))

print("\nArchivo guardado en:")
print(ruta)
