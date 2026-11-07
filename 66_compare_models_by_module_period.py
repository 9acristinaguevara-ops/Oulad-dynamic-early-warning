
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline


# 1. Cargar el dataset
ruta = "data/interim/week4_combined_features_v4.parquet"
df = pd.read_parquet(ruta)


# 2. Definir los periodos
periodos_entrenamiento = ["2013B", "2013J"]
periodos_prueba = ["2014B", "2014J"]


# 3. Definir las variables
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

# Modelo A: solo actividad
# Modelo B: actividad + evaluaciones
modelos = {
    "Solo actividad": variables_actividad,
    "Actividad + evaluaciones": variables_actividad + variables_evaluaciones
}


# 4. Preparar entrenamiento y prueba
train = df[df["code_presentation"].isin(periodos_entrenamiento)].copy()
test = df[df["code_presentation"].isin(periodos_prueba)].copy()

print("COMPARACIÓN DE MODELOS POR MÓDULO Y PERIODO")
print("Entrenamiento:", len(train))
print("Prueba:", len(test))


# 5. Entrenar y evaluar cada modelo
resultados = []

for nombre_modelo, variables in modelos.items():

    X_train = train[variables]
    y_train = train["target_future_withdrawal"]

    X_test = test[variables]
    y_test = test["target_future_withdrawal"]

    modelo = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clasificador", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    modelo.fit(X_train, y_train)

    probabilidades = modelo.predict_proba(X_test)[:, 1]
    predicciones = (probabilidades >= 0.5).astype(int)

    test_resultados = test.copy()
    test_resultados["probabilidad"] = probabilidades
    test_resultados["prediccion"] = predicciones

    print("\nMODELO:", nombre_modelo)

    for periodo in periodos_prueba:
        for modulo in sorted(test_resultados["code_module"].unique()):

            grupo = test_resultados[
                (test_resultados["code_presentation"] == periodo) &
                (test_resultados["code_module"] == modulo)
            ]

            if len(grupo) == 0:
                continue

            reales = grupo["target_future_withdrawal"]
            predichos = grupo["prediccion"]
            probabilidades_grupo = grupo["probabilidad"]

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
                auc = roc_auc_score(reales, probabilidades_grupo)
            else:
                auc = float("nan")

            alertas = int(predichos.sum())
            retiradas = int(reales.sum())

            print(
                "Periodo:", periodo,
                "| Modulo:", modulo,
                "| Estudiantes:", len(grupo),
                "| Retiradas:", retiradas,
                "| Alertas:", alertas,
                "| Precision:", round(precision, 3),
                "| Recall:", round(recall, 3),
                "| F1:", round(f1, 3),
                "| AUC:", round(auc, 3)
            )

            resultados.append({
                "modelo": nombre_modelo,
                "periodo": periodo,
                "modulo": modulo,
                "estudiantes": len(grupo),
                "retiradas": retiradas,
                "alertas": alertas,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "auc": auc
            })


# 6. Guardar resultados
resultados_df = pd.DataFrame(resultados)

ruta_salida = (
    "outputs/tables/"
    "model_comparison_by_module_period_v1.csv"
)

resultados_df.to_csv(ruta_salida, index=False)

print("\nArchivo guardado en:", ruta_salida)
print("\nSCRIPT FINALIZADO")
