
import pandas as pd

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
from pathlib import Path


# 1. Rutas
BASE = Path(".")
ENTRADA = BASE / "data" / "interim" / "week4_combined_features_v4.parquet"
SALIDA = BASE / "outputs" / "tables"

SALIDA.mkdir(parents=True, exist_ok=True)


# 2. Cargar datos
df = pd.read_parquet(ENTRADA)

print("Dataset cargado:", len(df), "filas")


# 3. Definir variables
actividad = [
    "week3_clicks",
    "week4_clicks",
    "week4_active_days",
    "week4_activity_records",
    "has_week4_activity",
    "weekly_change",
    "clics_recientes_3_4"
]

participacion = [
    "evaluaciones_programadas_34",
    "evaluaciones_entregadas_34"
]

modelos = {
    "A_solo_actividad": actividad,
    "B_actividad_participacion": actividad + participacion
}


# 4. Separar periodos
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


# 5. Buscar un umbral global usando solo validación
umbrales_posibles = [0.20, 0.30, 0.40, 0.50]

umbrales_elegidos = {}
modelos_entrenados = {}

for nombre, variables in modelos.items():

    modelo = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clasificador", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    modelo.fit(
        train_inicial[variables],
        train_inicial["target_future_withdrawal"]
    )

    probabilidades = modelo.predict_proba(
        validacion[variables]
    )[:, 1]

    mejor_umbral = 0.20
    mejor_f1 = -1

    for umbral in umbrales_posibles:

        predicciones = (
            probabilidades >= umbral
        ).astype(int)

        valor_f1 = f1_score(
            validacion["target_future_withdrawal"],
            predicciones,
            zero_division=0
        )

        if valor_f1 > mejor_f1:
            mejor_f1 = valor_f1
            mejor_umbral = umbral

    umbrales_elegidos[nombre] = mejor_umbral

    print(
        nombre,
        "| Umbral:", mejor_umbral,
        "| F1 validación:", round(mejor_f1, 3)
    )


# 6. Entrenar con ambos periodos de entrenamiento
resultados = []
predicciones_globales = []

for nombre, variables in modelos.items():

    modelo = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clasificador", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    modelo.fit(
        train_final[variables],
        train_final["target_future_withdrawal"]
    )

    umbral = umbrales_elegidos[nombre]

    for periodo in ["2014B", "2014J"]:

        grupo_periodo = test[
            test["code_presentation"] == periodo
        ].copy()

        if grupo_periodo.empty:
            continue

        probabilidades = modelo.predict_proba(
            grupo_periodo[variables]
        )[:, 1]

        predicciones = (
            probabilidades >= umbral
        ).astype(int)

        reales = grupo_periodo[
            "target_future_withdrawal"
        ].astype(int)

        # Métricas globales del periodo
        metricas = {
            "modelo": nombre,
            "periodo": periodo,
            "modulo": "TODOS",
            "umbral": umbral,
            "estudiantes": len(grupo_periodo),
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

        # Guardar predicciones para auditar los resultados
        predicciones_globales.append(pd.DataFrame({
            "modelo": nombre,
            "periodo": periodo,
            "code_module": grupo_periodo["code_module"].values,
            "id_student": grupo_periodo["id_student"].values,
            "real": reales.values,
            "probabilidad": probabilidades,
            "prediccion": predicciones,
            "umbral": umbral
        }))

        # Métricas por módulo
        for modulo in sorted(
            grupo_periodo["code_module"].unique()
        ):

            indices = (
                grupo_periodo["code_module"] == modulo
            ).values

            reales_modulo = reales.iloc[
                [i for i, valor in enumerate(indices) if valor]
            ]

            prob_modulo = probabilidades[indices]
            pred_modulo = predicciones[indices]

            resultados.append({
                "modelo": nombre,
                "periodo": periodo,
                "modulo": modulo,
                "umbral": umbral,
                "estudiantes": len(reales_modulo),
                "retiradas": int(reales_modulo.sum()),
                "alertas": int(pred_modulo.sum()),
                "precision": precision_score(
                    reales_modulo,
                    pred_modulo,
                    zero_division=0
                ),
                "recall": recall_score(
                    reales_modulo,
                    pred_modulo,
                    zero_division=0
                ),
                "f1": f1_score(
                    reales_modulo,
                    pred_modulo,
                    zero_division=0
                ),
                "accuracy": accuracy_score(
                    reales_modulo,
                    pred_modulo
                ),
                "auc": (
                    roc_auc_score(
                        reales_modulo, prob_modulo
                    )
                    if reales_modulo.nunique() == 2
                    else float("nan")
                )
            })


# 7. Guardar resultados detallados
tabla = pd.DataFrame(resultados)

archivo_detalle = SALIDA / "82_model_comparison_detailed.csv"
tabla.to_csv(archivo_detalle, index=False)

predicciones_df = pd.concat(
    predicciones_globales,
    ignore_index=True
)

archivo_predicciones = SALIDA / "82_model_comparison_predictions.csv"
predicciones_df.to_csv(archivo_predicciones, index=False)


# 8. Mostrar métricas globales
globales = tabla[tabla["modulo"] == "TODOS"].copy()

print("\nRESULTADOS GLOBALES")
print(
    globales[
        [
            "modelo", "periodo", "umbral", "estudiantes",
            "retiradas", "alertas", "precision", "recall",
            "f1", "accuracy", "auc"
        ]
    ].round(3).to_string(index=False)
)

print("\nRESULTADOS POR MÓDULO")
print(
    tabla[tabla["modulo"] != "TODOS"][
        [
            "modelo", "periodo", "modulo", "estudiantes",
            "retiradas", "alertas", "precision",
            "recall", "f1", "auc"
        ]
    ].round(3).to_string(index=False)
)

print("\nArchivos guardados:")
print(archivo_detalle)
print(archivo_predicciones)
