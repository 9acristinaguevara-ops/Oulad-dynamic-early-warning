
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, brier_score_loss

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

umbrales = [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.45, 0.50]

resultados = []

# 5. ANALIZAR CADA MODELO

for nombre, variables_modelo in configuraciones.items():

    print("\n" + "=" * 65)
    print(nombre)

    modelo = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    X_train = train[variables_modelo].fillna(0)
    y_train = train[objetivo]

    modelo.fit(X_train, y_train)

    # Calibración

    X_cal = calibracion[variables_modelo].fillna(0)
    y_cal = calibracion[objetivo]

    prob_cal = modelo.predict_proba(X_cal)[:, 1]

    calibrador = LogisticRegression(max_iter=1000)

    calibrador.fit(
        prob_cal.reshape(-1, 1),
        y_cal
    )

    # Selección del umbral

    X_umbral = seleccion_umbral[variables_modelo].fillna(0)
    y_umbral = seleccion_umbral[objetivo]

    prob_original_umbral = modelo.predict_proba(X_umbral)[:, 1]

    prob_cal_umbral = calibrador.predict_proba(
        prob_original_umbral.reshape(-1, 1)
    )[:, 1]

    mejor_umbral = None
    mejor_f1 = -1

    for umbral in umbrales:

        pred = (prob_cal_umbral >= umbral).astype(int)

        f1 = f1_score(
            y_umbral,
            pred,
            zero_division=0
        )

        if f1 > mejor_f1:
            mejor_f1 = f1
            mejor_umbral = umbral

    print("Umbral elegido:", mejor_umbral)
    print("F1 en selección:", round(mejor_f1, 4))

    # Evaluación de los periodos de prueba

    for periodo, datos in [
        ("2014B", prueba_b),
        ("2014J", prueba_j)
    ]:

        X_test = datos[variables_modelo].fillna(0)
        y_test = datos[objetivo]

        prob_original = modelo.predict_proba(X_test)[:, 1]

        prob_calibrada = calibrador.predict_proba(
            prob_original.reshape(-1, 1)
        )[:, 1]

        print("\nPeriodo:", periodo)
        print("Probabilidad mínima:", round(prob_calibrada.min(), 4))
        print("Percentil 25:", round(pd.Series(prob_calibrada).quantile(0.25), 4))
        print("Mediana:", round(pd.Series(prob_calibrada).median(), 4))
        print("Percentil 75:", round(pd.Series(prob_calibrada).quantile(0.75), 4))
        print("Probabilidad máxima:", round(prob_calibrada.max(), 4))

        auc = roc_auc_score(y_test, prob_calibrada)
        brier = brier_score_loss(y_test, prob_calibrada)

        # Revisar diferentes umbrales

        for umbral in umbrales:

            predicciones = (prob_calibrada >= umbral).astype(int)

            resultados.append({
                "modelo": nombre,
                "periodo": periodo,
                "umbral_elegido_2013J": mejor_umbral,
                "umbral_evaluado": umbral,
                "AUC": auc,
                "Brier": brier,
                "F1": f1_score(
                    y_test, predicciones, zero_division=0
                ),
                "precision": (
                    (y_test[predicciones == 1] == 1).mean()
                    if predicciones.sum() > 0 else 0
                ),
                "recall": (
                    ((y_test == 1) & (predicciones == 1)).sum()
                    / (y_test == 1).sum()
                ),
                "alertas": int(predicciones.sum()),
                "porcentaje_alertas": round(
                    predicciones.mean() * 100, 2
                )
            })

# 6. GUARDAR RESULTADOS

tabla = pd.DataFrame(resultados)

ruta = salida / "115_diagnostico_probabilidades.csv"

tabla.to_csv(ruta, index=False)

print("\nRESULTADOS DE UMBRALES")
print(tabla.round(4).to_string(index=False))

print("\nArchivo guardado en:")
print(ruta)
