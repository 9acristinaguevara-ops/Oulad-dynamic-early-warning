
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# Cargar datos
datos = pd.read_parquet(
    "data/interim/week4_combined_features_v4.parquet"
)

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

objetivo = "target_future_withdrawal"

# Separar entrenamiento y validación por cohorte
entrenamiento = datos[
    datos["code_presentation"] == "2013B"
].copy()

validacion = datos[
    datos["code_presentation"] == "2013J"
].copy()

# Preparar variables y objetivo
X_train = entrenamiento[variables]
y_train = entrenamiento[objetivo].astype(int)

X_val = validacion[variables]
y_val = validacion[objetivo].astype(int)

# Crear el modelo
modelo = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clasificador", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

# Entrenar solo con 2013B
modelo.fit(X_train, y_train)

# Evaluar entrenamiento y validación
prob_train = modelo.predict_proba(X_train)[:, 1]
prob_val = modelo.predict_proba(X_val)[:, 1]

umbral = 0.20

pred_train = (prob_train >= umbral).astype(int)
pred_val = (prob_val >= umbral).astype(int)

resultados = []

for nombre, reales, probabilidades, predicciones in [
    ("Entrenamiento_2013B", y_train, prob_train, pred_train),
    ("Validacion_2013J", y_val, prob_val, pred_val)
]:
    resultados.append({
        "conjunto": nombre,
        "estudiantes": len(reales),
        "tasa_retirada": reales.mean(),
        "AUC": roc_auc_score(reales, probabilidades),
        "F1": f1_score(reales, predicciones, zero_division=0),
        "precision": precision_score(
            reales, predicciones, zero_division=0
        ),
        "recall": recall_score(
            reales, predicciones, zero_division=0
        )
    })

resumen = pd.DataFrame(resultados)

print("\nDIAGNÓSTICO DE SOBREAJUSTE")
print(resumen.round(4).to_string(index=False))

resumen.to_csv(
    "outputs/tables/98_diagnostico_sobreajuste.csv",
    index=False
)

print("\nArchivo guardado:")
print("outputs/tables/98_diagnostico_sobreajuste.csv")
