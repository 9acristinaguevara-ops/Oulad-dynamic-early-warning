
import pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "data" / "interim" /
    "fixed_week4_dataset.parquet"
)

METRICS_FILE = (
    BASE_DIR / "outputs" / "tables" /
    "baseline_model_metrics.csv"
)

IMPORTANCE_FILE = (
    BASE_DIR / "outputs" / "tables" /
    "baseline_feature_importance.csv"
)

REPORT_FILE = (
    BASE_DIR / "outputs" / "logs" /
    "baseline_model_report.txt"
)

print("Cargando conjunto de datos...")
df = pd.read_parquet(INPUT_FILE)

# Variables que utilizaremos como predictores
features = [
    "weekly_clicks",
    "active_days",
    "activity_records",
    "has_previous_week",
    "weekly_change",
    "activity_decreased"
]

# Mantener solo las variables disponibles
features = [col for col in features if col in df.columns]

X = df[features].copy()
y = df["target_withdrawn"].copy()
groups = df["id_student"].copy()

# Dividir por estudiante para evitar que una misma persona
# aparezca en entrenamiento y prueba.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_index, test_index = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_index]
X_test = X.iloc[test_index]

y_train = y.iloc[train_index]
y_test = y.iloc[test_index]

train_students = set(groups.iloc[train_index])
test_students = set(groups.iloc[test_index])

print("\nTamaño de los conjuntos:")
print("Entrenamiento:", len(X_train))
print("Prueba:", len(X_test))
print("Estudiantes en entrenamiento:", len(train_students))
print("Estudiantes en prueba:", len(test_students))
print("Estudiantes compartidos:", len(train_students & test_students))

# Preparar el modelo:
# los valores ausentes se sustituyen por la mediana
# y el Random Forest aprende a clasificar.
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

print("\nEntrenando Random Forest...")
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

metrics = pd.DataFrame({
    "metrica": [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc"
    ],
    "valor": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
})

# Importancia de las variables
classifier = model.named_steps["classifier"]

importance = pd.DataFrame({
    "variable": features,
    "importancia": classifier.feature_importances_
})

importance = importance.sort_values(
    "importancia",
    ascending=False
)

# Crear carpetas y guardar resultados
METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
IMPORTANCE_FILE.parent.mkdir(parents=True, exist_ok=True)
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

metrics.to_csv(METRICS_FILE, index=False)
importance.to_csv(IMPORTANCE_FILE, index=False)

matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["No Withdrawn", "Withdrawn"],
    zero_division=0
)

with open(REPORT_FILE, "w", encoding="utf-8") as file:
    file.write("PRIMER MODELO: RANDOM FOREST\n\n")
    file.write("Punto de observación: semana 4\n")
    file.write("Separación: por identificador de estudiante\n\n")
    file.write("Métricas:\n")
    file.write(metrics.to_string(index=False))
    file.write("\n\nMatriz de confusión (filas reales, columnas predichas):\n")
    file.write(str(matrix))
    file.write("\n\nReporte de clasificación:\n")
    file.write(report)
    file.write("\n\nImportancia de variables:\n")
    file.write(importance.to_string(index=False))
    file.write(
        "\n\nAdvertencia: las importancias no demuestran causalidad. "
        "Este es un modelo inicial y no una validación definitiva "
        "de un sistema de alerta temprana.\n"
    )

print("\nRESULTADOS DEL MODELO")
print(metrics.to_string(index=False))

print("\nMatriz de confusión:")
print(matrix)

print("\nReporte de clasificación:")
print(report)

print("\nImportancia de las variables:")
print(importance.to_string(index=False))

print("\nArchivos guardados:")
print(METRICS_FILE)
print(IMPORTANCE_FILE)
print(REPORT_FILE)
