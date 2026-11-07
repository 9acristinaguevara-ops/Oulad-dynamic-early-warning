
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# 1. Cargar el dataset
df = pd.read_parquet(
    "data/interim/week4_combined_features.parquet"
)

# 2. Separar la variable objetivo
columnas_excluir = [
    "code_module",
    "code_presentation",
    "id_student",
    "date_registration",
    "date_unregistration",
    "final_result",
    "target_future_withdrawal"
]

X = df.drop(columns=columnas_excluir)
y = df["target_future_withdrawal"]
grupos = df["id_student"]

# 3. Separar entrenamiento y datos temporales
separacion1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=42
)

indices_train, indices_temporales = next(
    separacion1.split(X, y, groups=grupos)
)

X_train = X.iloc[indices_train]
y_train = y.iloc[indices_train]

X_temporales = X.iloc[indices_temporales]
y_temporales = y.iloc[indices_temporales]
grupos_temporales = grupos.iloc[indices_temporales]

# 4. Separar validación y prueba
separacion2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42
)

indices_validacion, indices_test = next(
    separacion2.split(
        X_temporales,
        y_temporales,
        groups=grupos_temporales
    )
)

X_validacion = X_temporales.iloc[indices_validacion]
y_validacion = y_temporales.iloc[indices_validacion]

X_test = X_temporales.iloc[indices_test]
y_test = y_temporales.iloc[indices_test]

print("Registros de entrenamiento:", len(X_train))
print("Registros de validación:", len(X_validacion))
print("Registros de prueba:", len(X_test))

# 5. Crear los modelos
modelos = {
    "Regresion Logistica": Pipeline([
        ("imputador", SimpleImputer(strategy="median")),
        ("escalador", StandardScaler()),
        ("modelo", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("imputador", SimpleImputer(strategy="median")),
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

umbrales = [0.20, 0.30, 0.40, 0.50]
resultados = []

# 6. Entrenar y evaluar
for nombre, modelo in modelos.items():

    modelo.fit(X_train, y_train)

    # Elegir el umbral usando validación
    probabilidades_validacion = modelo.predict_proba(
        X_validacion
    )[:, 1]

    for umbral in umbrales:

        predicciones = (
            probabilidades_validacion >= umbral
        ).astype(int)

        resultados.append({
            "modelo": nombre,
            "conjunto": "Validacion",
            "umbral": umbral,
            "accuracy": accuracy_score(
                y_validacion, predicciones
            ),
            "precision": precision_score(
                y_validacion, predicciones, zero_division=0
            ),
            "recall": recall_score(
                y_validacion, predicciones, zero_division=0
            ),
            "f1": f1_score(
                y_validacion, predicciones, zero_division=0
            ),
            "alertas": int(predicciones.sum())
        })

    # Evaluación final en prueba con umbral 0.5
    probabilidades_test = modelo.predict_proba(
        X_test
    )[:, 1]

    predicciones_test = (
        probabilidades_test >= 0.50
    ).astype(int)

    resultados.append({
        "modelo": nombre,
        "conjunto": "Prueba (umbral 0.5)",
        "umbral": 0.50,
        "accuracy": accuracy_score(
            y_test, predicciones_test
        ),
        "precision": precision_score(
            y_test, predicciones_test, zero_division=0
        ),
        "recall": recall_score(
            y_test, predicciones_test, zero_division=0
        ),
        "f1": f1_score(
            y_test, predicciones_test, zero_division=0
        ),
        "alertas": int(predicciones_test.sum())
    })

# 7. Mostrar y guardar los resultados
tabla = pd.DataFrame(resultados)

print("\nRESULTADOS DE LOS UMBRALES")
print(tabla.round(4).to_string(index=False))

tabla.to_csv(
    "outputs/tables/threshold_evaluation.csv",
    index=False
)

print("\nResultados guardados en:")
print("outputs/tables/threshold_evaluation.csv")
