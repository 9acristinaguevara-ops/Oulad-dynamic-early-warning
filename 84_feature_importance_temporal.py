
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score


# 1. Rutas
BASE = Path(".")
ENTRADA = BASE / "data" / "interim" / "week4_combined_features_v4.parquet"
SALIDA = BASE / "outputs" / "tables"

SALIDA.mkdir(parents=True, exist_ok=True)


# 2. Cargar datos
df = pd.read_parquet(ENTRADA)

train = df[df["code_presentation"] == "2013B"].copy()
validacion = df[df["code_presentation"] == "2013J"].copy()

print("Entrenamiento 2013B:", len(train))
print("Validación 2013J:", len(validacion))


# 3. Seleccionar variables del modelo B
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

X_train = train[variables]
y_train = train["target_future_withdrawal"]

X_validacion = validacion[variables]
y_validacion = validacion["target_future_withdrawal"]


# 4. Crear y entrenar el modelo
modelo = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clasificador", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

modelo.fit(X_train, y_train)


# 5. Calcular AUC de referencia
probabilidades = modelo.predict_proba(X_validacion)[:, 1]
auc_base = roc_auc_score(y_validacion, probabilidades)

print("\nAUC de validación:", round(auc_base, 4))


# 6. Importancia por permutación
resultado = permutation_importance(
    modelo,
    X_validacion,
    y_validacion,
    scoring="roc_auc",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

importancia = pd.DataFrame({
    "variable": variables,
    "importancia_media": resultado.importances_mean,
    "desviacion": resultado.importances_std
})

importancia = importancia.sort_values(
    "importancia_media",
    ascending=False
)

print("\nIMPORTANCIA POR PERMUTACIÓN EN 2013J")
print(importancia.round(4).to_string(index=False))


# 7. Guardar resultados
archivo = SALIDA / "84_importancia_temporal_model_b.csv"
importancia.to_csv(archivo, index=False)

print("\nResultado guardado en:")
print(archivo)
