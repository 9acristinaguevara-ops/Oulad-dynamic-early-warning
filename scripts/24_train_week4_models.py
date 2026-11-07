
import pandas as pd
from pathlib import Path

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

# Rutas
BASE = Path(".")
DATA = BASE / "data" / "interim" / "corrected_week4_final_dataset.parquet"
OUTPUTS = BASE / "outputs"
TABLES = OUTPUTS / "tables"
LOGS = OUTPUTS / "logs"

TABLES.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

# Leer conjunto corregido
datos = pd.read_parquet(DATA)

# Variables disponibles para la alerta
variables = [
    "week3_clicks",
    "week4_clicks",
    "week4_active_days",
    "week4_activity_records",
    "has_week4_activity",
    "weekly_change"
]

X = datos[variables].copy()
y = datos["target_future_withdrawal"]
grupos = datos["id_student"]

# Separar estudiantes para evitar que aparezcan en ambos conjuntos
separador = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

indices_entrenamiento, indices_prueba = next(
    separador.split(X, y, groups=grupos)
)

X_train = X.iloc[indices_entrenamiento]
X_test = X.iloc[indices_prueba]
y_train = y.iloc[indices_entrenamiento]
y_test = y.iloc[indices_prueba]

# Definir los modelos
logistica = Pipeline([
    ("imputador", SimpleImputer(strategy="median")),
    ("escalador", StandardScaler()),
    ("modelo", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

bosque = Pipeline([
    ("imputador", SimpleImputer(strategy="median")),
    ("modelo", RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

modelos = {
    "Regresion_Logistica": logistica,
    "Random_Forest": bosque
}

resultados = []

# Entrenar y evaluar
for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)

    predicciones = modelo.predict(X_test)
    probabilidades = modelo.predict_proba(X_test)[:, 1]

    resultados.append({
        "modelo": nombre,
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

# Guardar métricas
tabla = pd.DataFrame(resultados)
tabla.to_csv(
    TABLES / "week4_model_comparison.csv",
    index=False
)

# Reportar resultados
with open(
    LOGS / "week4_model_report.txt",
    "w",
    encoding="utf-8"
) as archivo:
    archivo.write("COMPARACION DE MODELOS - SEMANA 4\n")
    archivo.write(f"Total de registros: {len(datos)}\n")
    archivo.write(f"Entrenamiento: {len(indices_entrenamiento)}\n")
    archivo.write(f"Prueba: {len(indices_prueba)}\n")
    archivo.write(
        f"Estudiantes en entrenamiento: "
        f"{grupos.iloc[indices_entrenamiento].nunique()}\n"
    )
    archivo.write(
        f"Estudiantes en prueba: "
        f"{grupos.iloc[indices_prueba].nunique()}\n\n"
    )
    archivo.write(tabla.to_string(index=False))

print("COMPARACION TERMINADA")
print("\nTamaño de los conjuntos:")
print("Entrenamiento:", len(X_train))
print("Prueba:", len(X_test))

print("\nMétricas de evaluación:")
print(tabla.round(4).to_string(index=False))

print("\nResultados guardados en:")
print(TABLES / "week4_model_comparison.csv")
print(LOGS / "week4_model_report.txt")
