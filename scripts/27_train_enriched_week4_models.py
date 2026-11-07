
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

BASE = Path(".")
ENTRADA = BASE / "data" / "interim" / "week4_enriched_features.parquet"
SALIDA = BASE / "outputs" / "tables" / "enriched_week4_model_comparison.csv"

datos = pd.read_parquet(ENTRADA)

variables = [
    "week3_clicks",
    "week4_clicks",
    "week4_active_days",
    "week4_activity_records",
    "has_week4_activity",
    "weekly_change",
    "semanas_con_actividad",
    "clics_acumulados_0_4",
    "promedio_clics_0_4",
    "dias_activos_0_4",
    "clics_recientes_3_4",
    "clics_anteriores_0_2"
]

X = datos[variables]
y = datos["target_future_withdrawal"]
grupos = datos["id_student"]

separador = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

indices_train, indices_test = next(
    separador.split(X, y, groups=grupos)
)

X_train = X.iloc[indices_train]
X_test = X.iloc[indices_test]
y_train = y.iloc[indices_train]
y_test = y.iloc[indices_test]

grupos_train = set(grupos.iloc[indices_train])
grupos_test = set(grupos.iloc[indices_test])

print("TAMAÑO DE LOS DATOS")
print("Entrenamiento:", len(X_train))
print("Prueba:", len(X_test))
print("Estudiantes compartidos:",
      len(grupos_train.intersection(grupos_test)))

modelos = {
    "Regresion Logistica": Pipeline([
        ("imputador", SimpleImputer(strategy="median")),
        ("escalador", StandardScaler()),
        ("modelo", LogisticRegression(max_iter=1000))
    ]),
    "Random Forest": Pipeline([
        ("imputador", SimpleImputer(strategy="median")),
        ("modelo", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])
}

resultados = []

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

tabla = pd.DataFrame(resultados).round(4)

SALIDA.parent.mkdir(parents=True, exist_ok=True)
tabla.to_csv(SALIDA, index=False)

print("\nRESULTADOS DE LOS MODELOS MEJORADOS")
print(tabla.to_string(index=False))

print("\nArchivo guardado:", SALIDA)
