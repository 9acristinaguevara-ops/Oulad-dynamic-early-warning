
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# 1. RUTAS

BASE = Path(__file__).resolve().parent.parent

archivo = (
    BASE / "data" / "interim" /
    "week4_combined_features_v4.parquet"
)

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


# 4. DIVIDIR 2013J

calibracion, seleccion_umbral = train_test_split(
    validacion,
    test_size=0.5,
    random_state=42,
    stratify=validacion[objetivo]
)


# 5. PREPARAR LOS DATOS

X_train = train[variables].fillna(0)
y_train = train[objetivo]

X_cal = calibracion[variables].fillna(0)
y_cal = calibracion[objetivo]

X_umbral = seleccion_umbral[variables].fillna(0)
y_umbral = seleccion_umbral[objetivo]

pruebas = {
    "2014B": prueba_b,
    "2014J": prueba_j
}


# 6. ENTRENAR LA REGRESION LOGISTICA

modelo = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

modelo.fit(X_train, y_train)


# 7. CALIBRAR LAS PROBABILIDADES

prob_cal = modelo.predict_proba(X_cal)[:, 1]

calibrador = LogisticRegression(max_iter=1000)

calibrador.fit(
    prob_cal.reshape(-1, 1),
    y_cal
)


# 8. SELECCIONAR EL UMBRAL

prob_umbral_original = modelo.predict_proba(
    X_umbral
)[:, 1]

prob_umbral_cal = calibrador.predict_proba(
    prob_umbral_original.reshape(-1, 1)
)[:, 1]

umbrales = [
    0.10, 0.12, 0.15, 0.18,
    0.20, 0.22, 0.25, 0.30
]

mejor_umbral = umbrales[0]
mejor_f1 = -1

from sklearn.metrics import f1_score

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


print("Umbral seleccionado:", mejor_umbral)


# 9. COEFICIENTES ESTANDARIZADOS

# Las desviaciones se calculan únicamente con 2013B.

desviaciones = X_train.std(ddof=0)

coeficientes = modelo.coef_[0]

tabla_coeficientes = pd.DataFrame({
    "variable": variables,
    "coeficiente_original": coeficientes,
    "desviacion_entrenamiento": [
        desviaciones[v] for v in variables
    ],
    "coeficiente_por_desviacion": [
        coeficientes[i] * desviaciones[variables[i]]
        for i in range(len(variables))
    ]
})

tabla_coeficientes["direccion_asociacion"] = np.where(
    tabla_coeficientes["coeficiente_por_desviacion"] > 0,
    "Positiva",
    np.where(
        tabla_coeficientes["coeficiente_por_desviacion"] < 0,
        "Negativa",
        "Nula"
    )
)

ruta_coeficientes = (
    salida / "113_coeficientes_logistica.csv"
)

tabla_coeficientes.to_csv(
    ruta_coeficientes,
    index=False
)


# 10. IMPORTANCIA POR PERMUTACION

resultados_importancia = []

rng = np.random.default_rng(42)

for periodo, datos in pruebas.items():

    X_test = datos[variables].fillna(0)
    y_test = datos[objetivo]

    prob_original = modelo.predict_proba(
        X_test
    )[:, 1]

    prob_calibrada = calibrador.predict_proba(
        prob_original.reshape(-1, 1)
    )[:, 1]

    auc_original = roc_auc_score(
        y_test,
        prob_calibrada
    )

    for variable in variables:

        caidas_auc = []

        for repeticion in range(30):

            X_permutado = X_test.copy()

            valores = X_permutado[variable].to_numpy().copy()

            rng.shuffle(valores)

            X_permutado[variable] = valores

            prob_permutada = modelo.predict_proba(
                X_permutado
            )[:, 1]

            prob_permutada_cal = calibrador.predict_proba(
                prob_permutada.reshape(-1, 1)
            )[:, 1]

            auc_permutado = roc_auc_score(
                y_test,
                prob_permutada_cal
            )

            caida = auc_original - auc_permutado

            caidas_auc.append(caida)

        resultados_importancia.append({
            "periodo": periodo,
            "variable": variable,
            "auc_original": auc_original,
            "caida_auc_media": np.mean(caidas_auc),
            "desviacion_caida_auc": np.std(
                caidas_auc, ddof=1
            )
        })


# 11. GUARDAR RESULTADOS

tabla_importancia = pd.DataFrame(
    resultados_importancia
)

ruta_importancia = (
    salida / "113_importancia_permutacion_logistica.csv"
)

tabla_importancia.to_csv(
    ruta_importancia,
    index=False
)


# 12. MOSTRAR RESULTADOS

print("\nCOEFICIENTES DE LA REGRESION LOGISTICA")

print(
    tabla_coeficientes.round(5).to_string(index=False)
)

print("\nIMPORTANCIA POR PERMUTACION")

print(
    tabla_importancia.round(5).to_string(index=False)
)

print("\nARCHIVOS GUARDADOS:")
print(ruta_coeficientes)
print(ruta_importancia)
