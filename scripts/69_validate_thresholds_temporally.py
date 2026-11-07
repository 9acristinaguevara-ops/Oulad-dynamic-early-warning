
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

# 2. Separar entrenamiento, validacion y prueba
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

print("VALIDACION TEMPORAL DE UMBRALES")
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


# 4. Seleccionar umbrales usando solo 2013J
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

    X_train = train_inicial[variables]
    y_train = train_inicial["target_future_withdrawal"]

    X_val = validacion[variables]
    y_val = validacion["target_future_withdrawal"]

    modelo.fit(X_train, y_train)

    probabilidades = modelo.predict_proba(X_val)[:, 1]

    val_resultados = validacion[
        ["code_module", "target_future_withdrawal"]
    ].copy()

    val_resultados["probabilidad"] = probabilidades

    print("\nMODELO:", nombre_modelo)
    print("Seleccion de umbrales con 2013J")

    for modulo in sorted(val_resultados["code_module"].unique()):

        grupo = val_resultados[
            val_resultados["code_module"] == modulo
        ]

        reales = grupo["target_future_withdrawal"]

        if reales.nunique() < 2:
            print(
                "Modulo:", modulo,
                "- No se puede seleccionar umbral:",
                "solo existe una clase en validacion."
            )
            continue

        resultados_modulo = []

        for umbral in umbrales:

            predichos = (
                grupo["probabilidad"] >= umbral
            ).astype(int)

            f1 = f1_score(
                reales, predichos, zero_division=0
            )

            precision = precision_score(
                reales, predichos, zero_division=0
            )

            recall = recall_score(
                reales, predichos, zero_division=0
            )

            resultados_modulo.append({
                "umbral": umbral,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "alertas": int(predichos.sum())
            })

        tabla_modulo = pd.DataFrame(resultados_modulo)

        mejor = tabla_modulo.loc[
            tabla_modulo["f1"].idxmax()
        ]

        umbrales_elegidos[
            (nombre_modelo, modulo)
        ] = mejor["umbral"]

        print(
            "Modulo:", modulo,
            "| Umbral elegido:", mejor["umbral"],
            "| F1 validacion:", round(mejor["f1"], 3),
            "| Precision:", round(mejor["precision"], 3),
            "| Recall:", round(mejor["recall"], 3),
            "| Alertas:", int(mejor["alertas"])
        )

        resumen_validacion.append({
            "modelo": nombre_modelo,
            "modulo": modulo,
            "umbral_elegido": mejor["umbral"],
            "f1_validacion": mejor["f1"],
            "precision_validacion": mejor["precision"],
            "recall_validacion": mejor["recall"],
            "alertas_validacion": int(mejor["alertas"])
        })


# 5. Reentrenar con 2013B y 2013J y evaluar 2014
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

    probabilidades = modelo.predict_proba(
        test[variables]
    )[:, 1]

    test_resultados = test[
        ["code_presentation", "code_module",
         "target_future_withdrawal"]
    ].copy()

    test_resultados["probabilidad"] = probabilidades

    print("\nRESULTADOS FINALES:", nombre_modelo)

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

            clave = (nombre_modelo, modulo)

            # Si no se pudo seleccionar umbral para el modulo,
            # se usa 0.30 como alternativa previamente definida.
            umbral = umbrales_elegidos.get(clave, 0.30)

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


# 6. Guardar archivos sin sobrescribir resultados anteriores
tabla_validacion = pd.DataFrame(resumen_validacion)
tabla_prueba = pd.DataFrame(resultados_prueba)

tabla_validacion.to_csv(
    "outputs/tables/thresholds_selected_on_2013J_v1.csv",
    index=False
)

tabla_prueba.to_csv(
    "outputs/tables/temporal_validation_results_v1.csv",
    index=False
)

print("\nArchivos guardados:")
print("outputs/tables/thresholds_selected_on_2013J_v1.csv")
print("outputs/tables/temporal_validation_results_v1.csv")
print("\nSCRIPT FINALIZADO")
