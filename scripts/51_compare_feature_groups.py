
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

df = pd.read_parquet(
    "data/interim/week4_combined_features_v4.parquet"
)

# Variables que nunca deben usarse como predictores
excluir = [
    "code_module",
    "code_presentation",
    "id_student",
    "date_registration",
    "date_unregistration",
    "final_result",
    "target_future_withdrawal"
]

# Variables de actividad
actividad = [
    "semanas_con_actividad",
    "clics_acumulados_0_4",
    "promedio_clics_0_4",
    "dias_activos_0_4",
    "clics_recientes_3_4",
    "clics_anteriores_0_2",
    "has_week4_activity"
]

# Variables de entregas y evaluaciones
entregas = [
    "evaluaciones_programadas_34",
    "evaluaciones_entregadas_34",
    "proporcion_entregadas_34",
    "evaluaciones_aprobadas_34"
]

# Variable de calificaciones
notas = ["promedio_notas_34"]

experimentos = {
    "A_actividad": actividad,
    "B_actividad_entregas": actividad + entregas,
    "C_actividad_entregas_notas": actividad + entregas + notas
}

# Separar variables y objetivo
y = df["target_future_withdrawal"]
grupos = df["id_student"]

# Separar entrenamiento y prueba por estudiante
separador = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

indices_train, indices_test = next(
    separador.split(df, y, groups=grupos)
)

resultados = []

for nombre, variables in experimentos.items():

    X = df[variables]

    X_train = X.iloc[indices_train]
    X_test = X.iloc[indices_test]

    y_train = y.iloc[indices_train]
    y_test = y.iloc[indices_test]

    modelos = {
        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("modelo", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            ))
        ]),

        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("modelo", RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ))
        ])
    }

    for nombre_modelo, modelo in modelos.items():

        modelo.fit(X_train, y_train)

        predicciones = modelo.predict(X_test)
        probabilidades = modelo.predict_proba(X_test)[:, 1]

        resultados.append({
            "experimento": nombre,
            "modelo": nombre_modelo,
            "numero_variables": len(variables),
            "accuracy": accuracy_score(y_test, predicciones),
            "precision": precision_score(
                y_test, predicciones, zero_division=0
            ),
            "recall": recall_score(
                y_test, predicciones, zero_division=0
            ),
            "f1": f1_score(
                y_test, predicciones, zero_division=0
            ),
            "roc_auc": roc_auc_score(y_test, probabilidades)
        })

# Guardar resultados en un archivo nuevo
tabla = pd.DataFrame(resultados)

tabla.to_csv(
    "outputs/tables/feature_group_comparison_v1.csv",
    index=False
)

print("\nCOMPARACIÓN DE EXPERIMENTOS\n")
print(tabla.round(4).to_string(index=False))

print("\nResultados guardados en:")
print("outputs/tables/feature_group_comparison_v1.csv")

print("\nTamaño de los conjuntos:")
print("Entrenamiento:", len(indices_train))
print("Prueba:", len(indices_test))
print("Estudiantes compartidos entre conjuntos:",
      len(set(grupos.iloc[indices_train]) &
          set(grupos.iloc[indices_test])))
