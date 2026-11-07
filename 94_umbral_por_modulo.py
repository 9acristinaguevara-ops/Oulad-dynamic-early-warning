
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# 1. Rutas
ENTRADA = Path("data/interim/week4_combined_features_v4.parquet")
SALIDA = Path("outputs/tables")
SALIDA.mkdir(parents=True, exist_ok=True)

# 2. Variables del modelo B
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
umbrales = [0.20, 0.30, 0.40, 0.50]

# 3. Cargar datos
df = pd.read_parquet(ENTRADA)

entrenamiento = df[
    df["code_presentation"] == "2013B"
].copy()

validacion = df[
    df["code_presentation"] == "2013J"
].copy()

prueba = df[
    df["code_presentation"].isin(["2014B", "2014J"])
].copy()

# 4. Entrenar el modelo solo con 2013B
modelo = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clasificador", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

modelo.fit(
    entrenamiento[variables],
    entrenamiento[objetivo]
)

# 5. Calcular probabilidades
validacion["probabilidad"] = modelo.predict_proba(
    validacion[variables]
)[:, 1]

prueba["probabilidad"] = modelo.predict_proba(
    prueba[variables]
)[:, 1]

# 6. Seleccionar umbral por módulo usando 2013J
umbrales_modulo = {}
resultados_validacion = []

for modulo in sorted(validacion["code_module"].unique()):

    grupo = validacion[
        validacion["code_module"] == modulo
    ]

    reales = grupo[objetivo].astype(int)

    mejor_f1 = -1
    mejor_umbral = umbrales[0]

    for umbral in umbrales:

        predicciones = (
            grupo["probabilidad"] >= umbral
        ).astype(int)

        valor_f1 = f1_score(
            reales,
            predicciones,
            zero_division=0
        )

        if valor_f1 > mejor_f1:
            mejor_f1 = valor_f1
            mejor_umbral = umbral

    umbrales_modulo[modulo] = mejor_umbral

    resultados_validacion.append({
        "modulo": modulo,
        "umbral_elegido": mejor_umbral,
        "f1_validacion": mejor_f1,
        "estudiantes_validacion": len(grupo)
    })

# 7. Evaluar los umbrales en 2014B y 2014J
resultados_prueba = []

for periodo in ["2014B", "2014J"]:

    datos_periodo = prueba[
        prueba["code_presentation"] == periodo
    ]

    for modulo in sorted(datos_periodo["code_module"].unique()):

        grupo = datos_periodo[
            datos_periodo["code_module"] == modulo
        ]

        reales = grupo[objetivo].astype(int)
        probabilidades = grupo["probabilidad"]

        umbral_modulo = umbrales_modulo.get(
            modulo, 0.20
        )

        for tipo, umbral in [
            ("Global_020", 0.20),
            ("Por_modulo", umbral_modulo)
        ]:

            predicciones = (
                probabilidades >= umbral
            ).astype(int)

            resultados_prueba.append({
                "periodo": periodo,
                "modulo": modulo,
                "metodo": tipo,
                "umbral": umbral,
                "estudiantes": len(grupo),
                "retiradas_reales": int(reales.sum()),
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
                "auc": (
                    roc_auc_score(reales, probabilidades)
                    if reales.nunique() == 2
                    else float("nan")
                )
            })

# 8. Mostrar resultados
tabla_validacion = pd.DataFrame(resultados_validacion)
tabla_prueba = pd.DataFrame(resultados_prueba)

print("\nUMBRALES SELECCIONADOS EN 2013J")
print(tabla_validacion.round(3).to_string(index=False))

print("\nRESULTADOS POR MÓDULO EN 2014B Y 2014J")
print(tabla_prueba.round(3).to_string(index=False))

# 9. Comparación global ponderada por estudiantes
print("\nCOMPARACIÓN GLOBAL POR MÉTODO")

for metodo in ["Global_020", "Por_modulo"]:

    grupo = tabla_prueba[
        tabla_prueba["metodo"] == metodo
    ]

    for periodo in ["2014B", "2014J"]:

        datos = grupo[grupo["periodo"] == periodo]

        total_estudiantes = datos["estudiantes"].sum()
        total_alertas = datos["alertas"].sum()
        total_retiradas = datos["retiradas_reales"].sum()

        # Métricas globales calculadas a partir de las predicciones
        # individuales para evitar promediar métricas de módulos.
        filas = prueba[
            prueba["code_presentation"] == periodo
        ].copy()

        filas["prediccion"] = 0

        for _, fila in datos.iterrows():

            mascara = (
                (filas["code_module"] == fila["modulo"])
            )

            if metodo == "Global_020":
                umbral = 0.20
            else:
                umbral = umbrales_modulo.get(fila["modulo"], 0.20)

            filas.loc[mascara, "prediccion"] = (
                filas.loc[mascara, "probabilidad"] >= umbral
            ).astype(int)

        reales = filas[objetivo].astype(int)
        predicciones = filas["prediccion"]

        print(
            metodo,
            periodo,
            "| estudiantes:", total_estudiantes,
            "| retiradas:", total_retiradas,
            "| alertas:", total_alertas,
            "| precisión:", round(
                precision_score(reales, predicciones, zero_division=0), 3
            ),
            "| recall:", round(
                recall_score(reales, predicciones, zero_division=0), 3
            ),
            "| F1:", round(
                f1_score(reales, predicciones, zero_division=0), 3
            )
        )

# 10. Guardar archivos
tabla_validacion.to_csv(
    SALIDA / "94_umbrales_validacion.csv",
    index=False
)

tabla_prueba.to_csv(
    SALIDA / "94_comparacion_umbral_modulo.csv",
    index=False
)

print("\nArchivos guardados en:", SALIDA)
