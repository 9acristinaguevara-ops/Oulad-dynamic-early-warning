
import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score
)
from sklearn.pipeline import Pipeline


# 1. Rutas
ENTRADA = Path("data/interim/week4_combined_features_v4.parquet")
SALIDA = Path("outputs/tables")
SALIDA.mkdir(parents=True, exist_ok=True)


# 2. Cargar datos
df = pd.read_parquet(ENTRADA)

print("Dataset cargado:", len(df), "filas")


# 3. Variables del modelo B
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

# Solo estas variables se convertirán a cero
variables_semana4 = [
    "week4_clicks",
    "week4_active_days",
    "week4_activity_records"
]


# 4. Separar los periodos
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


# 5. Crear dos versiones del conjunto de datos
datos_originales = df.copy()
datos_corregidos = df.copy()

# Sustituir por cero únicamente los faltantes
# de las variables de conteo de la semana 4
datos_corregidos[variables_semana4] = (
    datos_corregidos[variables_semana4].fillna(0)
)


# 6. Comparar las dos versiones
versiones = {
    "Original": datos_originales,
    "Semana4_cero": datos_corregidos
}

resultados = []
predicciones_guardadas = []

umbrales_posibles = [0.20, 0.30, 0.40, 0.50]


for nombre_version, datos in versiones.items():

    print("\n==============================")
    print("VERSIÓN:", nombre_version)
    print("==============================")

    train_a = datos[
        datos["code_presentation"] == "2013B"
    ]

    valid_a = datos[
        datos["code_presentation"] == "2013J"
    ]

    train_f = datos[
        datos["code_presentation"].isin(["2013B", "2013J"])
    ]

    test_f = datos[
        datos["code_presentation"].isin(["2014B", "2014J"])
    ]

    # 7. Entrenar con 2013B y elegir umbral en 2013J
    modelo_validacion = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clasificador", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    modelo_validacion.fit(
        train_a[variables],
        train_a["target_future_withdrawal"]
    )

    probabilidades_validacion = modelo_validacion.predict_proba(
        valid_a[variables]
    )[:, 1]

    mejor_umbral = 0.20
    mejor_f1 = -1

    for umbral in umbrales_posibles:

        pred_validacion = (
            probabilidades_validacion >= umbral
        ).astype(int)

        valor_f1 = f1_score(
            valid_a["target_future_withdrawal"],
            pred_validacion,
            zero_division=0
        )

        if valor_f1 > mejor_f1:
            mejor_f1 = valor_f1
            mejor_umbral = umbral

    print("Umbral seleccionado:", mejor_umbral)
    print("F1 de validación:", round(mejor_f1, 3))

    # 8. Entrenar el modelo final con 2013B + 2013J
    modelo_final = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clasificador", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    modelo_final.fit(
        train_f[variables],
        train_f["target_future_withdrawal"]
    )

    # 9. Evaluar en los periodos de prueba
    for periodo in ["2014B", "2014J"]:

        grupo = test_f[
            test_f["code_presentation"] == periodo
        ].copy()

        if grupo.empty:
            continue

        probabilidades = modelo_final.predict_proba(
            grupo[variables]
        )[:, 1]

        predicciones = (
            probabilidades >= mejor_umbral
        ).astype(int)

        reales = grupo["target_future_withdrawal"].astype(int)

        metricas = {
            "version": nombre_version,
            "periodo": periodo,
            "umbral": mejor_umbral,
            "estudiantes": len(grupo),
            "retiradas": int(reales.sum()),
            "alertas": int(predicciones.sum()),
            "precision": precision_score(
                reales, predicciones, zero_division=0
            ),
            "recall": recall_score(
                reales, predicciones, zero_division=0
            ),
            "f1": f1_score(
                reales, predicciones, zero_division=0
            ),
            "accuracy": accuracy_score(reales, predicciones),
            "auc": (
                roc_auc_score(reales, probabilidades)
                if reales.nunique() == 2
                else float("nan")
            )
        }

        resultados.append(metricas)

        predicciones_guardadas.append(pd.DataFrame({
            "version": nombre_version,
            "periodo": periodo,
            "code_module": grupo["code_module"].values,
            "id_student": grupo["id_student"].values,
            "real": reales.values,
            "probabilidad": probabilidades,
            "prediccion": predicciones,
            "umbral": mejor_umbral
        }))


# 10. Guardar resultados
tabla = pd.DataFrame(resultados)

archivo_resultados = SALIDA / "89_comparacion_faltantes_resultados.csv"
tabla.to_csv(archivo_resultados, index=False)

predicciones_df = pd.concat(
    predicciones_guardadas,
    ignore_index=True
)

archivo_predicciones = SALIDA / "89_comparacion_faltantes_predicciones.csv"
predicciones_df.to_csv(archivo_predicciones, index=False)


# 11. Mostrar comparación
print("\nRESULTADOS DE PRUEBA")

columnas = [
    "version",
    "periodo",
    "umbral",
    "estudiantes",
    "retiradas",
    "alertas",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "auc"
]

print(tabla[columnas].round(3).to_string(index=False))

print("\nArchivos guardados:")
print(archivo_resultados)
print(archivo_predicciones)
