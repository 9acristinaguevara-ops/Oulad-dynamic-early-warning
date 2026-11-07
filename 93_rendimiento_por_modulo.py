
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score
)


# 1. Rutas
ENTRADA = Path("data/interim/week4_combined_features_v4.parquet")
SALIDA = Path("outputs/tables")
SALIDA.mkdir(parents=True, exist_ok=True)


# 2. Cargar datos
df = pd.read_parquet(ENTRADA)

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


# 3. Separar entrenamiento y prueba
entrenamiento = df[
    df["code_presentation"].isin(["2013B", "2013J"])
].copy()

prueba = df[
    df["code_presentation"].isin(["2014B", "2014J"])
].copy()


# 4. Entrenar modelo B
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
prueba["probabilidad_retirada"] = modelo.predict_proba(
    prueba[variables]
)[:, 1]

umbral = 0.20

prueba["prediccion"] = (
    prueba["probabilidad_retirada"] >= umbral
).astype(int)


# 6. Evaluar por módulo y período
resultados = []

for periodo in ["2014B", "2014J"]:

    datos_periodo = prueba[
        prueba["code_presentation"] == periodo
    ]

    for modulo in sorted(datos_periodo["code_module"].unique()):

        grupo = datos_periodo[
            datos_periodo["code_module"] == modulo
        ]

        reales = grupo[objetivo].astype(int)
        predicciones = grupo["prediccion"]
        probabilidades = grupo["probabilidad_retirada"]

        if reales.nunique() == 2:
            auc = roc_auc_score(reales, probabilidades)
        else:
            auc = float("nan")

        resultados.append({
            "periodo": periodo,
            "modulo": modulo,
            "estudiantes": len(grupo),
            "retiradas_reales": int(reales.sum()),
            "tasa_retirada": reales.mean(),
            "alertas": int(predicciones.sum()),
            "porcentaje_alertado": predicciones.mean(),
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
            "auc": auc
        })


# 7. Mostrar y guardar
tabla = pd.DataFrame(resultados)

print("\nRENDIMIENTO POR MÓDULO Y PERÍODO")
print(tabla.round(3).to_string(index=False))

archivo = SALIDA / "93_rendimiento_por_modulo.csv"
tabla.to_csv(archivo, index=False)

print("\nResultados guardados en:")
print(archivo)


# 8. Resumen de alertas por período
print("\nRESUMEN POR PERÍODO")

resumen = tabla.groupby("periodo").agg(
    estudiantes=("estudiantes", "sum"),
    retiradas=("retiradas_reales", "sum"),
    alertas=("alertas", "sum"),
    precision_media=("precision", "mean"),
    recall_medio=("recall", "mean"),
    f1_medio=("f1", "mean"),
    auc_medio=("auc", "mean")
)

print(resumen.round(3).to_string())
