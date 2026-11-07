
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.pipeline import Pipeline


# 1. Cargar datos
df = pd.read_parquet(
    "data/interim/week4_combined_features_v4.parquet"
)

# 2. Separar los periodos
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

print("VALIDACION TEMPORAL CON UMBRAL GLOBAL")
print("Entrenamiento inicial:", len(train_inicial))
print("Validacion:", len(validacion))
print("Entrenamiento final:", len(train_final))
print("Prueba:", len(test))


# 3. Definir variables
variables_actividad = [
    "week3_clicks",
    "week4_clicks",
    "week4_active_days",
    "week4_activity_records",
    "has_week4_activity",
    "weekly_change",
    "clics_recientes_3_4"
]

variables_evaluaciones = [
    "evaluaciones_programadas_34",
    "evaluaciones_entregadas_34",
    "promedio_notas_34",
    "evaluaciones_aprobadas_34"
]

modelos = {
    "Solo actividad": variables_actividad,
    "Actividad + evaluaciones": (
        variables_actividad + variables_evaluaciones
    )
}

umbrales = [0.20, 0.30, 0.40, 0.50]
periodos_prueba = ["2014B", "2014J"]


# 4. Entrenar y seleccionar umbrales en 2013J
umbrales_elegidos = {}
resumen_validacion = []

for nombre_modelo, variables in modelos.items():

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

    val_resultados = validacion[
        ["code_module", "target_future_withdrawal"]
    ].copy()

    val_resultados["probabilidad"] = probabilidades

    # Seleccionar un umbral global con toda la validacion
    resultados_globales = []

    for umbral in umbrales:

        predichos = (
            val_resultados["probabilidad"] >= umbral
        ).astype(int)

        f1 = f1_score(
            val_resultados["target_future_withdrawal"],
            predichos,
            zero_division=0
        )

        resultados_globales.append({
            "umbral": umbral,
            "f1": f1
        })

    tabla_global = pd.DataFrame(resultados_globales)

    mejor_global = tabla_global.loc[
        tabla_global["f1"].idxmax()
    ]

    umbral_global = mejor_global["umbral"]

    print("\nMODELO:", nombre_modelo)
    print(
        "Umbral global de validacion:",
        umbral_global,
        "| F1 global:",
        round(mejor_global["f1"], 3)
    )

    # Seleccionar umbral por modulo cuando sea posible
    for modulo in sorted(validacion["code_module"].unique()):

        grupo = val_resultados[
            val_resultados["code_module"] == modulo
        ]

        reales = grupo["target_future_withdrawal"]

        if reales.nunique() < 2:
            print(
                "Modulo:", modulo,
                "- Sin dos clases en validacion;",
                "se usara el umbral global."
            )

            umbral = umbral_global

        else:
            resultados_modulo = []

            for valor_umbral in umbrales:

                predichos = (
                    grupo["probabilidad"] >= valor_umbral
                ).astype(int)

                f1 = f1_score(
                    reales,
                    predichos,
                    zero_division=0
                )

                resultados_modulo.append({
                    "umbral": valor_umbral,
                    "f1": f1
                })

            tabla_modulo = pd.DataFrame(
                resultados_modulo
            )

            mejor = tabla_modulo.loc[
                tabla_modulo["f1"].idxmax()
            ]

            umbral = mejor["umbral"]

        umbrales_elegidos[
            (nombre_modelo, modulo)
        ] = umbral

        predichos = (
            grupo["probabilidad"] >= umbral
        ).astype(int)

        precision = precision_score(
            reales, predichos, zero_division=0
        )

        recall = recall_score(
            reales, predichos, zero_division=0
        )

        f1_final = f1_score(
            reales, predichos, zero_division=0
        )

        print(
            "Modulo:", modulo,
            "| Umbral seleccionado:", umbral,
            "| F1 validacion:", round(f1_final, 3)
        )

        resumen_validacion.append({
            "modelo": nombre_modelo,
            "modulo": modulo,
            "umbral_global": umbral_global,
            "umbral_elegido": umbral,
            "f1_validacion": f1_final,
            "precision_validacion": precision,
            "recall_validacion": recall
        })


# 5. Reentrenar y evaluar en 2014B y 2014J
resultados_prueba = []

for nombre_modelo, variables in modelos.items():

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

    test_resultados = test[
        [
            "code_presentation",
            "code_module",
            "target_future_withdrawal"
        ]
    ].copy()

    test_resultados["probabilidad"] = (
        modelo.predict_proba(test[variables])[:, 1]
    )

    print("\nRESULTADOS DE PRUEBA:", nombre_modelo)

    for periodo in periodos_prueba:

        for modulo in sorted(
            test_resultados["code_module"].unique()
        ):

            grupo = test_resultados[
                (test_resultados["code_presentation"] == periodo) &
                (test_resultados["code_module"] == modulo)
            ].copy()

            if len(grupo) == 0:
                continue

            umbral = umbrales_elegidos.get(
                (nombre_modelo, modulo),
                umbral_global
            )

            reales = grupo["target_future_withdrawal"]

            predichos = (
                grupo["probabilidad"] >= umbral
            ).astype(int)

            precision = precision_score(
                reales, predichos, zero_division=0
            )

            recall = recall_score(
                reales, predichos, zero_division=0
            )

            f1 = f1_score(
                reales, predichos, zero_division=0
            )

            if reales.nunique() == 2:
                auc = roc_auc_score(
                    reales, grupo["probabilidad"]
                )
            else:
                auc = float("nan")

            alertas = int(predichos.sum())
            retiradas = int(reales.sum())

            print(
                "Periodo:", periodo,
                "| Modulo:", modulo,
                "| Umbral:", umbral,
                "| Alertas:", alertas,
                "| Retiradas:", retiradas,
                "| Precision:", round(precision, 3),
                "| Recall:", round(recall, 3),
                "| F1:", round(f1, 3),
                "| AUC:", round(auc, 3)
            )

            resultados_prueba.append({
                "modelo": nombre_modelo,
                "periodo": periodo,
                "modulo": modulo,
                "umbral": umbral,
                "estudiantes": len(grupo),
                "retiradas": retiradas,
                "alertas": alertas,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "auc": auc
            })


# 6. Guardar resultados
pd.DataFrame(resumen_validacion).to_csv(
    "outputs/tables/global_fallback_thresholds_v1.csv",
    index=False
)

pd.DataFrame(resultados_prueba).to_csv(
    "outputs/tables/global_fallback_test_results_v1.csv",
    index=False
)

print("\nArchivos guardados:")
print("outputs/tables/global_fallback_thresholds_v1.csv")
print("outputs/tables/global_fallback_test_results_v1.csv")
print("SCRIPT FINALIZADO")
