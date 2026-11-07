
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score


# 1. Rutas
ENTRADA = Path("data/interim/week4_combined_features_v4.parquet")
SALIDA = Path("outputs/tables")

SALIDA.mkdir(parents=True, exist_ok=True)


# 2. Cargar datos
df = pd.read_parquet(ENTRADA)

periodos = ["2013B", "2013J"]

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

datos = df[df["code_presentation"].isin(periodos)].copy()


# 3. Comparar tamaño y abandono por periodo
resumen = datos.groupby("code_presentation").agg(
    estudiantes=("target_future_withdrawal", "size"),
    abandonos=("target_future_withdrawal", "sum"),
    tasa_abandono=("target_future_withdrawal", "mean")
).reset_index()

resumen["tasa_abandono"] = resumen["tasa_abandono"] * 100

print("\nRESUMEN DE LOS PERIODOS")
print(resumen.round(2).to_string(index=False))


# 4. Comparar las variables
comparacion = datos.groupby("code_presentation")[variables].mean().T

comparacion.columns = ["2013B", "2013J"]
comparacion["diferencia_2013J_menos_2013B"] = (
    comparacion["2013J"] - comparacion["2013B"]
)

print("\nPROMEDIO DE LAS VARIABLES POR PERIODO")
print(comparacion.round(3).to_string())


# 5. Entrenar con 2013B
train = datos[datos["code_presentation"] == "2013B"]
validacion = datos[datos["code_presentation"] == "2013J"]

X_train = train[variables]
y_train = train["target_future_withdrawal"]

X_validacion = validacion[variables]
y_validacion = validacion["target_future_withdrawal"]

modelo = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clasificador", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

modelo.fit(X_train, y_train)


# 6. Evaluar en ambos periodos
resultados = []

for nombre, X, y in [
    ("2013B_entrenamiento", X_train, y_train),
    ("2013J_validacion", X_validacion, y_validacion)
]:
    probabilidades = modelo.predict_proba(X)[:, 1]
    predicciones = (probabilidades >= 0.20).astype(int)

    resultados.append({
        "periodo": nombre,
        "estudiantes": len(y),
        "tasa_abandono": y.mean(),
        "AUC": roc_auc_score(y, probabilidades),
        "F1_umbral_020": f1_score(
            y, predicciones, zero_division=0
        )
    })

metricas = pd.DataFrame(resultados)
metricas["tasa_abandono"] = metricas["tasa_abandono"] * 100

print("\nCOMPARACIÓN DEL RENDIMIENTO")
print(metricas.round(4).to_string(index=False))


# 7. Guardar resultados
resumen.to_csv(
    SALIDA / "85_resumen_periodos.csv", index=False
)

comparacion.to_csv(
    SALIDA / "85_comparacion_variables.csv"
)

metricas.to_csv(
    SALIDA / "85_metricas_temporales.csv", index=False
)

print("\nArchivos guardados en outputs/tables:")
print("85_resumen_periodos.csv")
print("85_comparacion_variables.csv")
print("85_metricas_temporales.csv")
