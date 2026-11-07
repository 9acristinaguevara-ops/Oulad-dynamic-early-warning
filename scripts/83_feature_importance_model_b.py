
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance


# 1. Rutas
BASE = Path(".")
ENTRADA = BASE / "data" / "interim" / "week4_combined_features_v4.parquet"
SALIDA = BASE / "outputs" / "tables"

SALIDA.mkdir(parents=True, exist_ok=True)


# 2. Cargar los datos
df = pd.read_parquet(ENTRADA)

# Usar solamente los periodos de entrenamiento
train = df[
    df["code_presentation"].isin(["2013B", "2013J"])
].copy()


# 3. Definir variables del modelo B
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

X = train[variables]
y = train["target_future_withdrawal"]


# 4. Crear y entrenar el modelo
modelo = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clasificador", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

modelo.fit(X, y)


# 5. Importancia interna del Random Forest
clasificador = modelo.named_steps["clasificador"]

importancia_interna = pd.DataFrame({
    "variable": variables,
    "importancia": clasificador.feature_importances_
})

importancia_interna = importancia_interna.sort_values(
    "importancia",
    ascending=False
)

print("\nIMPORTANCIA INTERNA DEL RANDOM FOREST")
print(importancia_interna.round(4).to_string(index=False))


# 6. Importancia por permutación en validación
validacion = df[
    df["code_presentation"] == "2013J"
].copy()

X_validacion = validacion[variables]
y_validacion = validacion["target_future_withdrawal"]

resultado = permutation_importance(
    modelo,
    X_validacion,
    y_validacion,
    scoring="roc_auc",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

importancia_permutacion = pd.DataFrame({
    "variable": variables,
    "importancia_media": resultado.importances_mean,
    "desviacion": resultado.importances_std
})

importancia_permutacion = importancia_permutacion.sort_values(
    "importancia_media",
    ascending=False
)

print("\nIMPORTANCIA POR PERMUTACION EN VALIDACION")
print(importancia_permutacion.round(4).to_string(index=False))


# 7. Guardar resultados
archivo_interno = SALIDA / "83_importancia_interna_model_b.csv"
archivo_permutacion = SALIDA / "83_importancia_permutacion_model_b.csv"

importancia_interna.to_csv(archivo_interno, index=False)
importancia_permutacion.to_csv(archivo_permutacion, index=False)

print("\nArchivos guardados:")
print(archivo_interno)
print(archivo_permutacion)
